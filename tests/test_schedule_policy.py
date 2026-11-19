import unittest
from dataclasses import dataclass, field
from minisgl.scheduler.policy import SchedulePolicy


@dataclass
class FakePrefillManager:
    runnable: bool = True
    can_allocate: bool = True
    budgets_received: list[int] = field(default_factory=list)

    def schedule_next_batch(self, prefill_budget: int):
        if not self.runnable:
            return None
        self.budgets_received.append(prefill_budget)
        return "prefill_batch" if self.can_allocate else None


@dataclass
class FakeDecodeManager:
    runnable: bool = True

    def schedule_next_batch(self):
        return "decode_batch" if self.runnable else None


class TestSchedulePolicy(unittest.TestCase):
    def test_prefill_first_priority_and_kv_fallback(self):
        policy = SchedulePolicy("prefill_first", prefill_budget=8192)
        pm, dm = FakePrefillManager(), FakeDecodeManager()
        self.assertEqual(policy.select_next_batch(pm, dm), "prefill_batch")
        self.assertEqual(pm.budgets_received, [8192])

        # KV capacity insufficient -> fallback to decode
        pm.can_allocate = False
        self.assertEqual(policy.select_next_batch(pm, dm), "decode_batch")

    def test_decode_first_priority(self):
        policy = SchedulePolicy("decode_first", prefill_budget=8192)
        pm, dm = FakePrefillManager(), FakeDecodeManager()
        self.assertEqual(policy.select_next_batch(pm, dm), "decode_batch")

        dm.runnable = False
        self.assertEqual(policy.select_next_batch(pm, dm), "prefill_batch")

    def test_balanced_consecutive_decode_boundary_and_budget(self):
        policy = SchedulePolicy(
            "balanced",
            prefill_budget=8192,
            max_consecutive_decode=2,
            balanced_prefill_budget=512,
        )
        pm, dm = FakePrefillManager(), FakeDecodeManager()

        # 2 consecutive decode steps, then 1 bounded prefill step, then decode again
        self.assertEqual(policy.select_next_batch(pm, dm), "decode_batch")
        self.assertEqual(policy.consecutive_decode_steps, 1)
        self.assertEqual(policy.select_next_batch(pm, dm), "decode_batch")
        self.assertEqual(policy.consecutive_decode_steps, 2)
        self.assertEqual(policy.select_next_batch(pm, dm), "prefill_batch")
        self.assertEqual(policy.consecutive_decode_steps, 0)
        self.assertEqual(pm.budgets_received, [512])
        self.assertEqual(policy.select_next_batch(pm, dm), "decode_batch")

    def test_balanced_kv_fallback_and_single_queue(self):
        policy = SchedulePolicy(
            "balanced",
            prefill_budget=8192,
            max_consecutive_decode=1,
            balanced_prefill_budget=1024,
        )
        pm, dm = FakePrefillManager(can_allocate=False), FakeDecodeManager()

        # Step 1: decode; Step 2: prefill fails (KV full) -> fallback to decode
        self.assertEqual(policy.select_next_batch(pm, dm), "decode_batch")
        self.assertEqual(policy.select_next_batch(pm, dm), "decode_batch")
        self.assertEqual(pm.budgets_received, [1024])

        # Only prefill runnable -> uses full prefill_budget
        dm.runnable = False
        pm.can_allocate = True
        self.assertEqual(policy.select_next_batch(pm, dm), "prefill_batch")
        self.assertEqual(pm.budgets_received[-1], 8192)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            SchedulePolicy("unknown")
        with self.assertRaises(ValueError):
            SchedulePolicy("balanced", max_consecutive_decode=-1)
        with self.assertRaises(ValueError):
            SchedulePolicy("balanced", balanced_prefill_budget=0)


if __name__ == "__main__":
    unittest.main()
