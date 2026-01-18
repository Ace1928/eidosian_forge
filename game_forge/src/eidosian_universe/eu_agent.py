"""Agent definition for the Eidosian Universe."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import hypot
from typing import List, Tuple


@dataclass
class Agent:
    """Single agent within the Eidosian Universe."""

    x: float
    y: float
    dx: float
    dy: float
    energy: float
    mass: float
    components: List[Tuple[float, float]]
    color: Tuple[int, int, int]

    def update(
        self,
        config: "UniverseConfig",
        resources: List["Resource"],
        metabolism: "MetabolismRule",
    ) -> None:
        """Update agent position, energy, and interact with resources."""
        self.x += self.dx
        self.y += self.dy
        move_cost = hypot(self.dx, self.dy) * self.mass * metabolism.movement_cost
        self.energy -= config.energy_decay + move_cost
        for res in resources:
            if res.value > 0 and hypot(self.x - res.x, self.y - res.y) < config.agent_size * 2:
                gained = min(res.value, metabolism.max_energy - self.energy)
                self.energy += gained
                res.value -= gained

        # Bounce on boundaries
        if self.x < 0 or self.x > config.width:
            self.dx *= -1
        if self.y < 0 or self.y > config.height:
            self.dy *= -1

    def can_reproduce(self, config: "UniverseConfig") -> bool:
        """Return True if agent has enough energy to reproduce."""
        return self.energy >= config.reproduction_energy + config.reproduction_cost

    def reproduce(self, config: "UniverseConfig") -> "Agent":
        """Create a mutated child agent."""
        self.energy -= config.reproduction_cost
        child_energy: float = config.reproduction_cost / 2
        return Agent(
            x=self.x + random.uniform(-10, 10),
            y=self.y + random.uniform(-10, 10),
            dx=random.uniform(-1, 1),
            dy=random.uniform(-1, 1),
            energy=child_energy,
            mass=max(0.5, self.mass + random.uniform(-0.1, 0.1)),
            components=[(x + random.uniform(-0.2, 0.2), y + random.uniform(-0.2, 0.2)) for x, y in self.components],
            color=tuple(min(255, max(0, c + random.randint(-20, 20))) for c in self.color),
        )
