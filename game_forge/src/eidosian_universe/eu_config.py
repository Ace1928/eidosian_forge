"""Configuration for the Eidosian Universe simulation."""

from dataclasses import dataclass

@dataclass
class UniverseConfig:
    """Settings controlling universe simulation parameters."""

    width: int = 1024
    height: int = 768
    fullscreen: bool = True
    background_color: tuple[int, int, int] = (10, 10, 30)
    agent_count: int = 50
    max_agents: int = 200
    agent_size: int = 4
    energy_decay: float = 0.1
    movement_cost: float = 0.05
    maintenance_cost: float = 0.02
    reproduction_energy: float = 80.0
    reproduction_cost: float = 30.0
    starting_energy: float = 50.0
    max_energy: float = 100.0
    temperature: float = 1.0
    resource_count: int = 200
    resource_value: float = 25.0
