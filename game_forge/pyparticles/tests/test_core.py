import pytest
from pyparticles.core.types import SimulationConfig, ParticleState, ForceType, RenderMode

def test_config_defaults():
    cfg = SimulationConfig.default()
    # V6.1: MASSIVELY expanded world for emergent dynamics
    assert cfg.width == 1400
    assert cfg.height == 1000
    assert cfg.world_size == 100.0  # 10x larger world
    assert cfg.dt == 0.005
    assert cfg.friction == 0.5  # Increased for stability
    assert cfg.angular_friction == 2.0
    assert cfg.thermostat_enabled == True
    assert cfg.substeps == 2
    assert cfg.wave_repulsion_strength == 5.0  # Moderate for subtle effect
    assert cfg.num_particles == 10000
    assert cfg.num_types == 16
    assert cfg.max_velocity == 10.0  # Energy capping

def test_config_validation():
    cfg = SimulationConfig.default()
    cfg.dt = 0.05  # Too large
    cfg.num_particles = 200000  # Exceeds max
    warnings = cfg.validate()
    assert len(warnings) >= 2  # dt warning + clamping warning
    assert cfg.num_particles == cfg.max_particles  # Should be clamped

def test_config_presets():
    small = SimulationConfig.small_world()
    assert small.world_size == 20.0  # V6.1: increased
    assert small.num_particles == 1000
    
    large = SimulationConfig.large_world()
    assert large.world_size == 200.0  # V6.1: much larger
    assert large.num_particles == 30000
    
    # V6.1: New huge preset
    huge = SimulationConfig.huge_world()
    assert huge.world_size == 500.0
    assert huge.num_particles == 50000

def test_particle_state_allocation():
    max_p = 100
    state = ParticleState.allocate(max_p)
    assert state.pos.shape == (max_p, 2)
    assert state.colors.shape == (max_p,)
    assert state.angle.shape == (max_p,)
    assert state.ang_vel.shape == (max_p,)
    assert state.active == 0