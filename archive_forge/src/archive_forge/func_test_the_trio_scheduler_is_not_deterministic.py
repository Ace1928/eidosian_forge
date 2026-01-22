from __future__ import annotations
from typing import TYPE_CHECKING
import trio
def test_the_trio_scheduler_is_not_deterministic() -> None:
    traces = []
    for _ in range(10):
        traces.append(trio.run(scheduler_trace))
    assert len(set(traces)) == len(traces)