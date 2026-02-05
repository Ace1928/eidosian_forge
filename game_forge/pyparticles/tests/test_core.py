import pytest
from pyparticles.core.types import SimulationConfig, ParticleState, ForceType, RenderMode

def test_config_defaults():
    cfg = SimulationConfig.default()
    assert cfg.width == 1200
    assert cfg.dt == 0.005  # V6: Reduced for stability
    assert cfg.friction == 0.5
    assert cfg.angular_friction == 2.0
    assert cfg.thermostat_enabled == True
    assert cfg.substeps == 1
    assert cfg.wave_repulsion_strength == 50.0

def test_config_validation():
    cfg = SimulationConfig.default()
    cfg.dt = 0.05  # Too large
    cfg.num_particles = 100000  # Exceeds max
    warnings = cfg.validate()
    assert len(warnings) >= 2
    assert cfg.num_particles == cfg.max_particles  # Should be clamped

def test_particle_state_allocation():
    max_p = 100
    state = ParticleState.allocate(max_p)
    assert state.pos.shape == (max_p, 2)
    assert state.colors.shape == (max_p,)
    assert state.angle.shape == (max_p,)
    assert state.ang_vel.shape == (max_p,)
    assert state.active == 0