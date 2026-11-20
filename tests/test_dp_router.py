import hashlib
import unittest
from scripts.dp_router import DPRouter


def _find_payload_for_backend(target_idx: int, n: int, prefix_chars: int = 512) -> dict:
    for i in range(10000):
        prompt = f"prefix-{i}-" + ("x" * prefix_chars)
        digest = int(hashlib.blake2b(prompt[:prefix_chars].encode("utf-8"), digest_size=8).hexdigest(), 16)
        if digest % n == target_idx:
            return {"prompt": prompt}
    raise RuntimeError("No matching payload found")


class TestDPRouter(unittest.TestCase):
    def test_prefix_hash_keeps_affinity_within_skew(self):
        router = DPRouter(["http://b0", "http://b1"], policy="prefix_hash", max_skew=2)
        p0 = _find_payload_for_backend(0, 2)
        router.in_flight = [2, 0]  # diff == 2, not > max_skew(2)
        self.assertEqual(router.select_backend(p0), 0)
        self.assertEqual(router.skew_fallbacks, 0)

    def test_prefix_hash_falls_back_when_skew_exceeded(self):
        router = DPRouter(["http://b0", "http://b1"], policy="prefix_hash", max_skew=2)
        p0 = _find_payload_for_backend(0, 2)
        router.in_flight = [3, 0]  # diff == 3 > max_skew(2) -> fall back to backend 1
        self.assertEqual(router.select_backend(p0), 1)
        self.assertEqual(router.skew_fallbacks, 1)


if __name__ == "__main__":
    unittest.main()
