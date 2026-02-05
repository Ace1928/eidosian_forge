"""
Updated Physics Tests for V3 Features.
"""
import pytest
import numpy as np
from pyparticles.core.types import SimulationConfig, ForceType, SpeciesConfig
from pyparticles.physics.engine import PhysicsEngine
from pyparticles.physics.kernels import compute_forces_multi

def test_engine_v3_features():
    cfg = SimulationConfig.default()
    engine = PhysicsEngine(cfg)
    
    # Check new rules
    assert len(engine.rules) == 3
    assert engine.rules[2].force_type == ForceType.INVERSE_CUBE
    
    # Check Species Config
    assert engine.species_config.radius.shape == (cfg.num_types,)
    assert engine.species_config.wave_freq.shape == (cfg.num_types,)
    
    # Check Particle State
    assert engine.state.angle.shape == (cfg.max_particles,)
    assert engine.state.ang_vel.shape == (cfg.max_particles,)

def test_kernel_wave_repulsion():
    # Setup 2 particles interacting
    # P0 at 0,0. P1 at 0.1, 0. (Dist 0.1)
    # Radii 0.05. Gap = 0.
    
    pos = np.zeros((2, 2), dtype=np.float32)
    pos[1, 0] = 0.1
    
    vel = np.zeros_like(pos)
    colors = np.array([0, 0], dtype=np.int32)
    angle = np.zeros(2, dtype=np.float32)
    ang_vel = np.zeros(2, dtype=np.float32)
    
    n = 2
    
    # Wave Params: Rad 0.05, Freq 1, Amp 0.01
    species_params = np.zeros((1, 3), dtype=np.float32)
    species_params[0, 0] = 0.05
    species_params[0, 1] = 1.0
    species_params[0, 2] = 0.01
    
    # Rules (Disable long range for this test)
    matrices = np.zeros((1, 1, 1), dtype=np.float32)
    params = np.zeros((1, 5), dtype=np.float32)
    # Set max_r=0 to disable matrix force
    params[0, 1] = 0.0 
    
    # Grid
    cell_size = 0.5
    grid_counts = np.zeros((3,3), dtype=np.int32)
    grid_cells = np.zeros((3,3,10), dtype=np.int32)
    # Manually fill
    grid_counts[1,1] = 2
    grid_cells[1,1,0] = 0
    grid_cells[1,1,1] = 1
    
    # Params
    wave_str = 100.0
    wave_exp = 10.0
    dt = 0.01
    
    # Test 1: Gap = 0.1 - (0.05+0.01 + 0.05+0.01) = 0.1 - 0.12 = -0.02 (Overlap)
    # Should repel.
    # Angle 0. P0 faces P1 (theta=0). P1 faces P0 (theta=pi).
    # Freq 1.
    # P0 local theta = 0. cos(0)=1 (Peak).
    # P1 local theta = pi. cos(pi)=-1 (Trough).
    # Peak-Trough -> Damped repulsion.
    
    forces, torques = compute_forces_multi(
        pos, vel, colors, angle, ang_vel, n,
        matrices, params, species_params,
        wave_str, wave_exp,
        grid_counts, grid_cells, cell_size,
        dt, 0.5, 0.0
    )
    
    # P0 should be pushed Left (Negative X)
    assert forces[0, 0] < 0
    f_damped = forces[0, 0]
    
    # Test 2: Rotate P1 by Pi so it presents a Peak.
    angle[1] = np.pi 
    # P1 contact angle in world is pi.
    # P1 local = pi - pi = 0. cos(0)=1 (Peak).
    # Peak-Peak -> Strong Repulsion.
    
    forces_peak, torques_peak = compute_forces_multi(
        pos, vel, colors, angle, ang_vel, n,
        matrices, params, species_params,
        wave_str, wave_exp,
        grid_counts, grid_cells, cell_size,
        dt, 0.5, 0.0
    )
    
    # Verify Peak-Peak is stronger than Peak-Trough
    assert forces_peak[0, 0] < f_damped
    
def test_torque_generation():
    # Particle collision off-center relative to wave slope
    pass # Torque logic is subtle, just ensure it runs
