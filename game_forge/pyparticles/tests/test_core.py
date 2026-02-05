import pytest
from pyparticles.core.types import SimulationConfig, ParticleState, ForceType, RenderMode

def test_config_defaults():
    cfg = SimulationConfig.default()
    assert cfg.width == 1200
    assert cfg.dt == 0.01 # Updated default
    assert cfg.wave_repulsion_strength == 50.0

def test_particle_state_allocation():
    max_p = 100
    state = ParticleState.allocate(max_p)
    assert state.pos.shape == (max_p, 2)
    assert state.colors.shape == (max_p,)
    assert state.angle.shape == (max_p,)
    assert state.ang_vel.shape == (max_p,)
    assert state.active == 0