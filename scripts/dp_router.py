import argparse
import hashlib
import json
from typing import List

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


class DPRouter:
    def __init__(
        self,
        backends: List[str],
        policy: str,
        prefix_chars: int = 512,
        max_skew: int = 2,
        max_keepalive_connections: int = 0,
    ):
        if not backends:
            raise ValueError("backends must not be empty")
        if max_skew < 0:
            raise ValueError(f"max_skew must be >= 0, got {max_skew}")
        self.backends = [b.rstrip("/") for b in backends]
        self.policy = policy
        self.prefix_chars = prefix_chars
        self.max_skew = max_skew
        self.max_keepalive_connections = max_keepalive_connections
        self.rr_idx = 0
        self.in_flight = [0] * len(self.backends)
        self.total_routed = [0] * len(self.backends)
        self.skew_fallbacks = 0

    def _extract_prefix(self, payload: dict) -> str:
        if "messages" in payload and isinstance(payload["messages"], list):
            parts = []
            for m in payload["messages"]:
                content = m.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
            text = "\n".join(parts)
        else:
            text = str(payload.get("prompt", ""))
        return text[: self.prefix_chars]

    def _least_loaded_idx(self) -> int:
        n = len(self.backends)
        return min(range(n), key=lambda i: (self.in_flight[i], self.total_routed[i], i))

    def select_backend(self, payload: dict) -> int:
        n = len(self.backends)
        if n == 1:
            return 0
        if self.policy == "round_robin":
            idx = self.rr_idx % n
            self.rr_idx += 1
            return idx
        elif self.policy == "least_running":
            return self._least_loaded_idx()
        elif self.policy == "prefix_hash":
            prefix = self._extract_prefix(payload)
            digest = int(hashlib.blake2b(prefix.encode("utf-8"), digest_size=8).hexdigest(), 16)
            pref_idx = digest % n
            best_idx = self._least_loaded_idx()
            # If preferred backend is skewed beyond max_skew over the least-loaded worker, fall back
            if self.in_flight[pref_idx] - self.in_flight[best_idx] > self.max_skew:
                self.skew_fallbacks += 1
                return best_idx
            return pref_idx
        raise ValueError(f"Unknown routing policy: {self.policy}")


def create_app(router: DPRouter) -> FastAPI:
    app = FastAPI(title="Mini-SGLang DP Router")
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        limits=httpx.Limits(
            max_connections=512,
            max_keepalive_connections=router.max_keepalive_connections,
        ),
    )

    @app.on_event("shutdown")
    async def shutdown_event():
        await client.aclose()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "policy": router.policy,
            "max_skew": router.max_skew,
            "backends": router.backends,
            "in_flight": router.in_flight,
            "total_routed": router.total_routed,
            "skew_fallbacks": router.skew_fallbacks,
        }

    @app.post("/v1/{path:path}")
    async def proxy_v1(path: str, request: Request):
        body = await request.body()
        payload = json.loads(body) if body else {}
        is_stream = bool(payload.get("stream", False))

        idx = router.select_backend(payload)
        target_url = f"{router.backends[idx]}/v1/{path}"
        router.in_flight[idx] += 1
        router.total_routed[idx] += 1

        if is_stream:
            req = client.build_request(
                "POST",
                target_url,
                content=body,
                headers={"Content-Type": "application/json"},
            )
            try:
                resp = await client.send(req, stream=True)
            except Exception:
                router.in_flight[idx] -= 1
                raise

            async def stream_gen():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()
                    router.in_flight[idx] -= 1

            return StreamingResponse(
                stream_gen(),
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "text/event-stream"),
            )
        else:
            try:
                resp = await client.post(
                    target_url,
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            finally:
                router.in_flight[idx] -= 1

    return app


def main():
    parser = argparse.ArgumentParser(description="Async DP Router for Mini-SGLang")
    parser.add_argument("--backends", type=str, required=True, help="Comma-separated backend URLs")
    parser.add_argument(
        "--policy",
        type=str,
        default="least_running",
        choices=["round_robin", "least_running", "prefix_hash"],
    )
    parser.add_argument("--prefix-chars", type=int, default=512)
    parser.add_argument("--max-skew", type=int, default=2, help="Max inflight gap before prefix_hash falls back")
    parser.add_argument("--max-keepalive-connections", type=int, default=0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1918)
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    router = DPRouter(
        backends=backends,
        policy=args.policy,
        prefix_chars=args.prefix_chars,
        max_skew=args.max_skew,
        max_keepalive_connections=args.max_keepalive_connections,
    )
    app = create_app(router)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
