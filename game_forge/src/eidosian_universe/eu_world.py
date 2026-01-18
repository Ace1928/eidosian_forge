"""Core simulation logic for the Eidosian Universe."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import pygame

from .eu_agent import Agent, Component
from .eu_config import UniverseConfig
from .eu_rules import (
    GravityRule,
    MetabolismRule,
    ResourceRule,
    EnvironmentRule,
)


@dataclass
class Resource:
    """Simple energy source within the universe."""

    x: float
    y: float
    value: float


class Universe:
    """Simulates a universe populated by agents."""

    def __init__(self, config: UniverseConfig) -> None:
        self.config = config
        self.agents: List[Agent] = []
        self.resources: List[Resource] = []
        self.gravity = GravityRule()
        self.metabolism = MetabolismRule(
            movement_cost=config.movement_cost,
            reproduction_cost=config.reproduction_cost,
            maintenance_cost=config.maintenance_cost,
            max_energy=config.max_energy,
        )
        self.environment = EnvironmentRule(temperature=config.temperature)
        self.resource_rule = ResourceRule()
        self.log: List[str] = []
        self._initialize_agents()
        self._initialize_resources()

    def _initialize_agents(self) -> None:
        """Create initial population of agents."""
        for _ in range(self.config.agent_count):
            self.agents.append(self._create_random_agent())

    def _create_random_agent(self, *, x: float | None = None, y: float | None = None) -> Agent:
        """Generate a random agent."""
        return Agent(
            x=random.uniform(0, self.config.width) if x is None else x,
            y=random.uniform(0, self.config.height) if y is None else y,
            dx=random.uniform(-1, 1),
            dy=random.uniform(-1, 1),
            energy=self.config.starting_energy,
            mass=random.uniform(0.5, 2.0),
            components=[
                Component(
                    offset=(random.uniform(-1, 1), random.uniform(-1, 1)),
                    mass=random.uniform(0.1, 0.5),
                    efficiency=random.uniform(0.8, 1.2),
                )
                for _ in range(random.randint(1, 3))
            ],
            color=(
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
            ),
        )

    def _initialize_resources(self) -> None:
        for _ in range(self.config.resource_count):
            self.resources.append(
                Resource(
                    x=random.uniform(0, self.config.width),
                    y=random.uniform(0, self.config.height),
                    value=self.config.resource_value,
                )
            )

    def update(self) -> None:
        """Advance simulation state."""
        # slowly vary environment temperature
        self.environment.temperature = max(
            0.5,
            min(
                5.0,
                self.environment.temperature + random.uniform(-0.01, 0.01),
            ),
        )

        self.gravity.apply(self.agents)

        new_agents: List[Agent] = []
        for agent in list(self.agents):
            agent.update(self.config, self.environment, self.resources, self.metabolism)
            if agent.can_reproduce(self.config) and len(self.agents) < self.config.max_agents:
                new_agents.append(agent.reproduce(self.config))
            if agent.energy <= 0:
                self.agents.remove(agent)
        self.agents.extend(new_agents)

        # remove depleted resources
        self.resources = [r for r in self.resources if r.value > 0]
        # respawn resources as needed
        while len(self.resources) < self.config.resource_count:
            self.resources.append(
                Resource(
                    x=random.uniform(0, self.config.width),
                    y=random.uniform(0, self.config.height),
                    value=self.config.resource_value,
                )
            )

    def log_event(self, message: str) -> None:
        """Record an event in the universe log and print it."""
        self.log.append(message)
        print(message)

    def render(self, surface: pygame.Surface) -> None:
        """Draw all agents to the surface."""
        surface.fill(self.config.background_color)
        for res in self.resources:
            pygame.draw.circle(
                surface,
                (0, 200, 100),
                (int(res.x), int(res.y)),
                max(2, int(self.config.agent_size / 2)),
            )
        for agent in self.agents:
            pygame.draw.circle(
                surface,
                agent.color,
                (int(agent.x), int(agent.y)),
                self.config.agent_size,
            )
