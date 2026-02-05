"""
Updated Physics Tests for V3/V6 Features.
"""
import pytest
import numpy as np
from pyparticles.core.types import SimulationConfig, ForceType, SpeciesConfig
from pyparticles.physics.engine import PhysicsEngine
from pyparticles.physics.kernels import compute_forces_multi

def test_engine_v3_features():
    cfg = SimulationConfig.default()
    engine = PhysicsEngine(cfg)
    
    # Check rules: V6 has Linear, Gravity, Core Repulsion
    assert len(engine.rules) == 3
    assert engine.rules[0].force_type == ForceType.LINEAR
    assert engine.rules[1].force_type == ForceType.INVERSE_SQUARE  # Gravity
    assert engine.rules[2].force_type == ForceType.REPEL_ONLY  # Core Repulsion
    
    # Check Species Config - V6 has extended params
    assert engine.species_config.radius.shape == (cfg.num_types,)
    assert engine.species_config.wave_freq.shape == (cfg.num_types,)
    assert engine.species_config.spin_inertia.shape == (cfg.num_types,)
    assert engine.species_config.base_spin_rate.shape == (cfg.num_types,)
    
    # Check Particle State
    assert engine.state.angle.shape == (cfg.max_particles,)
    assert engine.state.ang_vel.shape == (cfg.max_particles,)

def test_kernel_wave_repulsion():
    # Setup 2 particles interacting
    # P0 at 0,0. P1 at 0.09, 0. (Dist 0.09)
    # Radii 0.05.
    
    pos = np.zeros((2, 2), dtype=np.float32)
    pos[1, 0] = 0.09 
    
    colors = np.array([0, 0], dtype=np.int32)
    angle = np.zeros(2, dtype=np.float32)
    n = 2
    
    # Wave Params: Rad 0.05, Freq 1, Amp 0.01 + spin params (V6: 6 columns)
    species_params = np.zeros((1, 6), dtype=np.float32)
    species_params[0, 0] = 0.05  # radius
    species_params[0, 1] = 1.0   # freq
    species_params[0, 2] = 0.01  # amp
    species_params[0, 3] = 1.0   # inertia
    species_params[0, 4] = 1.0   # spin_friction
    species_params[0, 5] = 0.0   # base_spin
    
    # Rules (Disable long range for this test)
    matrices = np.zeros((1, 1, 1), dtype=np.float32)
    params = np.zeros((1, 8), dtype=np.float32)  # V6: 8-column params
    # Set max_r=0 to disable matrix force
    params[0, 1] = 0.0 
    
    # Grid
    cell_size = 0.5
    grid_counts = np.zeros((4,4), dtype=np.int32)
    grid_cells = np.zeros((4,4,10), dtype=np.int32)
    # Manually fill at correct coords (cx=2, cy=2 for pos near 0 with offset 1 and cell 0.5)
    grid_counts[2,2] = 2
    grid_cells[2,2,0] = 0
    grid_cells[2,2,1] = 1
    
    # Params
    wave_str = 100.0
    wave_exp = 10.0
    
    # Test 1: Peak-Trough Interaction
    # Angle 0. P0 faces P1 (theta=0). P1 faces P0 (theta=pi).
    # P0 local theta = 0. cos(0)=1 (Peak).
    # P1 local theta = pi. cos(pi)=-1 (Trough).
    forces, torques = compute_forces_multi(
        pos, colors, angle, n,
        matrices, params, species_params,
        wave_str, wave_exp,
        grid_counts, grid_cells, cell_size,
        0.0, 1.0  # gravity, half_world
    )    
    
    # Force logic: P0 should be pushed left
    assert forces[0, 0] < 0.0
    assert forces[1, 0] > 0.0    
    
    f_damped = forces[0, 0]

    # Test 2: Rotate P1 by Pi so it presents a Peak.
    angle[1] = np.pi 
    # Peak-Peak -> Strong Repulsion.
    
    forces_peak, torques_peak = compute_forces_multi(
        pos, colors, angle, n,
        matrices, params, species_params,
        wave_str, wave_exp,
        grid_counts, grid_cells, cell_size,
        0.0, 1.0  # gravity, half_world
    )
    
    # Verify Peak-Peak is stronger than Peak-Trough
    assert forces_peak[0, 0] < f_damped

def test_torque_generation():
    # Setup particles such that waves are sloped at contact
    pos = np.zeros((2, 2), dtype=np.float32)
    pos[1, 0] = 0.09 
    
    colors = np.array([0, 0], dtype=np.int32)
    angle = np.array([0.1, 0.0], dtype=np.float32) # Slight tilt
    n = 2
    
    # V6: 6-column species params
    species_params = np.zeros((1, 6), dtype=np.float32)
    species_params[0, 0] = 0.05  # radius
    species_params[0, 1] = 4.0   # High freq for steeper slopes
    species_params[0, 2] = 0.02  # amp
    species_params[0, 3] = 1.0   # inertia
    species_params[0, 4] = 1.0   # spin_friction
    species_params[0, 5] = 0.0   # base_spin
    
    matrices = np.zeros((1, 1, 1), dtype=np.float32)
    params = np.zeros((1, 8), dtype=np.float32)  # V6: 8-column params
    params[0, 1] = 0.0 
    
    cell_size = 0.5
    grid_counts = np.zeros((4,4), dtype=np.int32)
    grid_cells = np.zeros((4,4,10), dtype=np.int32)
    grid_counts[2,2] = 2
    grid_cells[2,2,0] = 0
    grid_cells[2,2,1] = 1
    
    forces, torques = compute_forces_multi(
        pos, colors, angle, n,
        matrices, params, species_params,
        100.0, 10.0,
        grid_counts, grid_cells, cell_size,
        0.0, 1.0  # gravity, half_world
    )
    
    # Torque should be non-zero due to sloped wave interaction
    assert abs(torques[0]) > 0.0