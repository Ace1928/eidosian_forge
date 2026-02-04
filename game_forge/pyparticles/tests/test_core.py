import pytest
from pyparticles.core.types import SimulationConfig, ParticleState, RenderMode
import numpy as np

def test_config_defaults():
    cfg = SimulationConfig.default()
    assert cfg.width == 1200
    assert cfg.dt == 0.02
    assert cfg.render_mode == RenderMode.SPRITES

def test_particle_state_allocation():
    max_p = 100
    state = ParticleState.allocate(max_p)
    assert state.pos.shape == (max_p, 2)
    assert state.vel.shape == (max_p, 2)
    assert state.colors.shape == (max_p,)
    assert state.active == 0
