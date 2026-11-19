import argparse, asyncio, time
import numpy as np
from openai import AsyncOpenAI

def make_prompt(req_id: int, target_tokens: int) -> str:
    header = f"[REQ-{req_id}-SALT-{req_id * 7919}] Summarize:\n"
    unit = "distributed inference scheduling latency throughput batching memory "
    return header + (unit * max(1, (target_tokens - 20) // 8))

async def send_one(client, model, req_id, is_long, delay, max_tokens):
    await asyncio.sleep(delay)
    prompt = make_prompt(req_id, 8000 if is_long else 512)
    t0, first_t, prev_t = time.perf_counter(), None, None
    itls, out_tokens = [], 0
    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0, stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if not delta: continue
        now = time.perf_counter()
        if first_t is None: first_t = now
        else: itls.append((now - prev_t) * 1000.0)
        prev_t, out_tokens = now, out_tokens + 1
    ttft = (first_t - t0) * 1000.0 if first_t else 0.0
    return {"is_long": is_long, "ttft": ttft, "itls": itls, "tokens": out_tokens}

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--port", type=int, default=1919)
    p.add_argument("--model", default="/root/lanyun-tmp/models/Qwen3-4B")
    p.add_argument("--num-short", type=int, default=24)
    p.add_argument("--num-long", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--qps", type=float, default=1.5)
    args = p.parse_args()
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")
    plan, s, l = [], args.num_short, args.num_long
    while s > 0 or l > 0:
        for _ in range(4):
            if s > 0: plan.append(False); s -= 1
        if l > 0: plan.append(True); l -= 1
    t0 = time.perf_counter()
    res = await asyncio.gather(*[send_one(client, args.model, i, lg, i / args.qps, args.max_tokens) for i, lg in enumerate(plan)])
    dt = time.perf_counter() - t0
    p95 = lambda v: float(np.percentile(v, 95)) if v else 0.0
    tok = sum(r["tokens"] for r in res)
    print(f"\n=== [{args.tag}] Elapsed: {dt:.2f}s | Throughput: {tok / dt:.2f} tok/s ===")
    print(f"All   TTFT P95: {p95([r['ttft'] for r in res]):8.1f} ms | All   ITL P95: {p95([x for r in res for x in r['itls']]):6.2f} ms")
    s_itl = [x for r in res if not r["is_long"] for x in r["itls"]]
    p99 = lambda v: float(np.percentile(v, 99)) if v else 0.0
    mx = lambda v: float(max(v)) if v else 0.0
    print(f"Short TTFT P95: {p95([r['ttft'] for r in res if not r['is_long']]):8.1f} ms | Short ITL P95: {p95(s_itl):6.2f} ms | P99: {p99(s_itl):6.2f} ms | Max: {mx(s_itl):6.1f} ms")
    print(f"Long  TTFT P95: {p95([r['ttft'] for r in res if r['is_long']]):8.1f} ms")

if __name__ == "__main__":
    asyncio.run(main())
