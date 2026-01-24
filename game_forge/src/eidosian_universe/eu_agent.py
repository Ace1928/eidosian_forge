"""Agent definition for the Eidosian Universe."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import hypot
from typing import List, Tuple, TYPE_CHECKING

from .eu_config import UniverseConfig
from eidosian_core import eidosian

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .eu_rules import EnvironmentRule, MetabolismRule


@dataclass
class Component:
    """Sub-element of an agent providing mass and efficiency."""

    offset: Tuple[float, float]
    mass: float
    efficiency: float = 1.0

    @eidosian()
    def mutate(self) -> "Component":
        """Return a slightly mutated copy of this component."""
        ox, oy = self.offset
        return Component(
            offset=(ox + random.uniform(-0.2, 0.2), oy + random.uniform(-0.2, 0.2)),
            mass=max(0.1, self.mass + random.uniform(-0.05, 0.05)),
            efficiency=max(0.5, min(1.5, self.efficiency + random.uniform(-0.05, 0.05))),
        )


@dataclass
class Agent:
    """Single agent within the Eidosian Universe."""

    x: float
    y: float
    dx: float
    dy: float
    energy: float
    mass: float
    components: List[Component]
    color: Tuple[int, int, int]

    @eidosian()
    def total_mass(self) -> float:
        """Return aggregate mass including components."""
        return self.mass + sum(c.mass for c in self.components)

    @eidosian()
    def update(
        self,
        config: "UniverseConfig",
        environment: "EnvironmentRule",
        resources: List["Resource"],
        metabolism: "MetabolismRule",
    ) -> None:
        """Update agent position, energy, and interact with resources."""
        self.x += self.dx
        self.y += self.dy
        self.dx *= 1 - environment.friction
        self.dy *= 1 - environment.friction
        move_cost = hypot(self.dx, self.dy) * self.total_mass() * metabolism.movement_cost
        maintenance = (
            sum(c.mass * (1.0 - c.efficiency) for c in self.components)
            * metabolism.maintenance_cost
        )
        self.energy -= metabolism.energy_loss(environment.temperature) + move_cost + maintenance
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

    @eidosian()
    def can_reproduce(self, config: "UniverseConfig") -> bool:
        """Return True if agent has enough energy to reproduce."""
        return self.energy >= config.reproduction_energy + config.reproduction_cost

    @eidosian()
    def reproduce(self, config: "UniverseConfig") -> "Agent":
        """Create a mutated child agent."""
        self.energy -= config.reproduction_cost
        child_energy: float = config.reproduction_cost / 2
        child_components = [c.mutate() for c in self.components]
        if random.random() < 0.1 and len(child_components) < 5:
            child_components.append(
                Component(
                    offset=(random.uniform(-1, 1), random.uniform(-1, 1)),
                    mass=0.5,
                )
            )
        return Agent(
            x=self.x + random.uniform(-10, 10),
            y=self.y + random.uniform(-10, 10),
            dx=random.uniform(-1, 1),
            dy=random.uniform(-1, 1),
            energy=child_energy,
            mass=max(0.5, self.mass + random.uniform(-0.1, 0.1)),
            components=child_components,
            color=tuple(min(255, max(0, c + random.randint(-20, 20))) for c in self.color),
        )
