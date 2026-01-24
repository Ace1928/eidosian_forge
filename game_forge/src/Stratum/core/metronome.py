"""
Metronome schedules work for each simulation tick.

The Metronome slices the available compute budget across all
subsystems (Quanta, Chemistry, etc.) and determines how many
microticks are available to each active region. This component keeps
per‑tick statistics for profiling and can adjust level of detail (LOD)
when the workload exceeds the budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
from eidosian_core import eidosian


@dataclass
class MetronomeStats:
    tick: int
    microticks_used: int = 0
    active_cells: int = 0
    energy_total: float = 0.0


class Metronome:
    """Manage timing and compute budgets for each simulation tick."""

    def __init__(self, config: 'EngineConfig'):
        from .config import EngineConfig  # type: ignore
        self.cfg = config
        # maintain a simple record of stats per tick
        self.stats: MetronomeStats = MetronomeStats(tick=0)

    @eidosian()
    def allocate(self, tick: int) -> Dict[str, int]:
        """Allocate compute budgets for this tick.

        Currently returns a dictionary mapping subsystem names to integer
        budgets representing microtick counts. In this simplified
        prototype, only the Quanta subsystem uses a budget.
        """
        self.stats.tick = tick
        # for now allocate a fixed number of microticks across all active
        # regions based on config. Real implementation could use time
        # measurement to scale budgets dynamically.
        return {
            "quanta_micro_ops": self.cfg.grid_w * self.cfg.grid_h * self.cfg.microtick_cap_per_region
        }

    @eidosian()
    def record_stats(self, tick: int) -> None:
        """Placeholder to record per tick stats.

        In a full implementation this would aggregate counters from
        various subsystems. The prototype simply increments the tick
        number; see ``MetronomeStats`` for fields updated externally.
        """
        self.stats.tick = tick