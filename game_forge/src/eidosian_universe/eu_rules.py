"""Mutable rule definitions for the Eidosian Universe simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .eu_agent import Agent


@dataclass
class GravityRule:
    """Simple gravitational attraction between agents."""

    strength: float = 0.05
    max_range: float = 100.0

    def apply(self, agents: List[Agent]) -> None:
        """Modify agent velocities based on mutual attraction."""
        count = len(agents)
        for i in range(count):
            a = agents[i]
            for j in range(i + 1, count):
                b = agents[j]
                dx = b.x - a.x
                dy = b.y - a.y
                dist_sq = dx * dx + dy * dy
                if dist_sq == 0 or dist_sq > self.max_range ** 2:
                    continue
                dist = dist_sq ** 0.5
                force = self.strength / (dist_sq)
                ax = force * dx / dist
                ay = force * dy / dist
                a.dx += ax
                a.dy += ay
                b.dx -= ax
                b.dy -= ay


@dataclass
class MetabolismRule:
    """Parameters governing energy costs and limits."""

    movement_cost: float = 0.05
    reproduction_cost: float = 30.0
    max_energy: float = 100.0


@dataclass
class ResourceRule:
    """Parameters controlling resource spawning."""

    spawn_rate: int = 1

