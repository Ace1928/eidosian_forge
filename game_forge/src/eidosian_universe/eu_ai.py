"""Heuristic AI for refining universe rules during runtime."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING
from eidosian_core import eidosian

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from .eu_world import Universe


class EidosAI:
    """Simple heuristic controller adjusting universe rules."""

    def __init__(self, universe: Universe) -> None:
        self.universe = universe
        self._ticks = 0

    @eidosian()
    def update(self) -> None:
        """Potentially adjust rules based on universe state."""
        self._ticks += 1
        if self._ticks % 300 == 0:
            self._tweak_gravity()
        if self._ticks % 500 == 0:
            self._tweak_metabolism()
        if self._ticks % 400 == 0:
            self._tweak_environment()

    def _tweak_gravity(self) -> None:
        gravity = self.universe.gravity
        old = gravity.strength
        new = max(0.01, min(0.2, old * random.uniform(0.95, 1.05)))
        gravity.strength = new
        self.universe.log_event(
            f"Eidos refined gravity: {old:.3f} -> {new:.3f}"
        )

    def _tweak_metabolism(self) -> None:
        meta = self.universe.metabolism
        old_move = meta.movement_cost
        new_move = max(0.01, min(0.2, old_move * random.uniform(0.9, 1.1)))
        meta.movement_cost = new_move
        self.universe.log_event(
            f"Eidos adjusted movement cost: {old_move:.3f} -> {new_move:.3f}"
        )

    def _tweak_environment(self) -> None:
        env = self.universe.environment
        old_temp = env.temperature
        env.temperature = max(0.5, min(5.0, old_temp * random.uniform(0.98, 1.02)))
        self.universe.log_event(
            f"Eidos tuned temperature: {old_temp:.2f} -> {env.temperature:.2f}"
        )

