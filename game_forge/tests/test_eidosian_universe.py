import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame
pygame.init()

from eidosian_universe.eu_world import Universe, Resource
from eidosian_universe.eu_config import UniverseConfig


def test_agent_consumes_resource():
    config = UniverseConfig()
    uni = Universe(config)
    agent = uni.agents[0]
    res = Resource(x=agent.x, y=agent.y, value=10.0)
    uni.resources = [res]
    old_energy = agent.energy
    agent.update(config, uni.resources, uni.metabolism)
    assert agent.energy > old_energy
    assert uni.resources[0].value < 10.0


def test_agent_reproduction():
    config = UniverseConfig()
    uni = Universe(config)
    agent = uni.agents[0]
    agent.energy = config.reproduction_energy + config.reproduction_cost
    child = agent.reproduce(config)
    assert child is not None
    assert agent.energy == config.reproduction_energy
    assert child.energy == config.reproduction_cost / 2
