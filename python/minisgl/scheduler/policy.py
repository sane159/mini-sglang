from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisgl.core import Batch

    from .decode import DecodeManager
    from .prefill import PrefillManager


SUPPORTED_SCHEDULE_POLICIES = ("prefill_first", "decode_first", "balanced")


@dataclass
class SchedulePolicy:
    policy: str = "prefill_first"
    prefill_budget: int = 8192
    max_consecutive_decode: int = 8
    balanced_prefill_budget: int | None = None
    adaptive_budget: bool = False
    consecutive_decode_steps: int = 0

    def __post_init__(self) -> None:
        if self.policy not in SUPPORTED_SCHEDULE_POLICIES:
            raise ValueError(
                f"Unsupported schedule policy: {self.policy!r}. "
                f"Expected one of {SUPPORTED_SCHEDULE_POLICIES}."
            )
        if self.prefill_budget <= 0:
            raise ValueError(f"prefill_budget must be > 0, got {self.prefill_budget}")
        if self.max_consecutive_decode < 0:
            raise ValueError(
                f"max_consecutive_decode must be >= 0, got {self.max_consecutive_decode}"
            )
        if self.balanced_prefill_budget is not None and self.balanced_prefill_budget <= 0:
            raise ValueError(
                f"balanced_prefill_budget must be > 0, got {self.balanced_prefill_budget}"
            )

    @property
    def effective_balanced_budget(self) -> int:
        if self.balanced_prefill_budget is None:
            return self.prefill_budget
        return min(self.prefill_budget, self.balanced_prefill_budget)

    def _next_balanced_prefill_budget(self, prefill_manager: PrefillManager) -> int:
        base_budget = self.effective_balanced_budget
        if not self.adaptive_budget:
            return base_budget
        # If prefill queue has multiple pending requests, scale up budget up to prefill_budget
        # to drain queued TTFT backlog faster while keeping bounded bursts.
        pending = getattr(prefill_manager, "pending_list", None)
        pending_len = len(pending) if pending is not None else 0
        if pending_len >= 4:
            scaled = min(self.prefill_budget, max(base_budget, base_budget * 2))
            return scaled
        return base_budget

    def select_next_batch(
        self,
        prefill_manager: PrefillManager,
        decode_manager: DecodeManager,
    ) -> Batch | None:
        if self.policy == "prefill_first":
            return (
                prefill_manager.schedule_next_batch(self.prefill_budget)
                or decode_manager.schedule_next_batch()
            )

        if self.policy == "decode_first":
            return (
                decode_manager.schedule_next_batch()
                or prefill_manager.schedule_next_batch(self.prefill_budget)
            )

        # balanced policy
        has_decode = decode_manager.runnable
        has_prefill = prefill_manager.runnable

        if not has_decode and not has_prefill:
            self.consecutive_decode_steps = 0
            return None

        if not has_decode:
            self.consecutive_decode_steps = 0
            return prefill_manager.schedule_next_batch(self.prefill_budget)

        if not has_prefill:
            self.consecutive_decode_steps = 0
            return decode_manager.schedule_next_batch()

        # Both decode and prefill are runnable
        if self.consecutive_decode_steps < self.max_consecutive_decode:
            batch = decode_manager.schedule_next_batch()
            if batch is not None:
                self.consecutive_decode_steps += 1
                return batch

        # Reached consecutive decode limit: give queued prefill a chance with bounded/adaptive budget
        budget = self._next_balanced_prefill_budget(prefill_manager)
        batch = prefill_manager.schedule_next_batch(budget)
        if batch is not None:
            self.consecutive_decode_steps = 0
            return batch

        # Fallback to decode when KV cache / table capacity is insufficient for prefill
        return decode_manager.schedule_next_batch()
