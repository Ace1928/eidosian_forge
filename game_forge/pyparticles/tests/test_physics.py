import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pyparticles.core.types import SimulationConfig
from pyparticles.physics.engine import PhysicsEngine
from pyparticles.physics.kernels import compute_forces_multi

def test_engine_rules():
    cfg = SimulationConfig.default()
    engine = PhysicsEngine(cfg)
    assert len(engine.rules) == 3 # Linear, InvSq, Strong
    
    # Check packing
    mats, params = engine._pack_rules()
    assert mats.shape == (3, 6, 6)
    assert params.shape == (3, 5)

def test_kernel_compute_forces_multi_logic():
    # Setup 2 particles
    pos = np.array([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32)
    vel = np.zeros_like(pos)
    colors = np.array([0, 0], dtype=np.int32)
    angle = np.zeros(2, dtype=np.float32)
    ang_vel = np.zeros(2, dtype=np.float32)
    n = 2
    
    # Setup Rules
    # Rule 0: Linear Repulsion (dist < 0.1)
    mats = np.zeros((2, 1, 1), dtype=np.float32)
    mats[0, 0, 0] = 1.0 
    mats[1, 0, 0] = 0.5 
    
    params = np.zeros((2, 5), dtype=np.float32)
    # R0: Linear
    params[0, 0] = 0.02
    params[0, 1] = 0.1
    params[0, 2] = 1.0
    params[0, 3] = 0.0
    params[0, 4] = 0.0
    
    # R1: InvSq
    params[1, 0] = 0.01
    params[1, 1] = 0.5
    params[1, 2] = 1.0
    params[1, 3] = 0.05
    params[1, 4] = 1.0
    
    # Species Params (1 Type)
    # Rad, Freq, Amp
    species_params = np.array([[0.05, 1.0, 0.0]], dtype=np.float32)
    
    # Grid
    cell_size = 0.5
    grid_counts = np.zeros((4, 4), dtype=np.int32)
    grid_cells = np.zeros((4, 4, 10), dtype=np.int32)
    # Manually fill grid for kernel test? 
    # Or rely on kernel handling empty grid if not populated?
    # Actually compute_forces_multi assumes grid is populated.
    # We must populate it.
    cx0, cy0 = 2, 2 # 0.0 -> index 2 (since +1.0 offset / 0.5 = 2)
    grid_counts[cy0, cx0] = 2
    grid_cells[cy0, cx0, 0] = 0
    grid_cells[cy0, cx0, 1] = 1
    
    dt = 0.01
    fric = 1.0
    grav = 0.0
    wave_str = 0.0 # Disable wave force for this test
    wave_exp = 10.0
    
    forces, torques = compute_forces_multi(
        pos, vel, colors, angle, ang_vel, n,
        mats, params, species_params,
        wave_str, wave_exp,
        grid_counts, grid_cells, cell_size,
        dt, fric, grav
    )
    
    # Dist = 0.05. Linear (0.02 < 0.05 < 0.1) -> Attraction.
    # InvSq -> Attraction.
    # P0 should be pulled +X
    assert forces[0, 0] > 0

def test_kernel_repel_only():
    # Test Type 2
    pos = np.array([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32)
    vel = np.zeros_like(pos)
    colors = np.array([0, 0], dtype=np.int32)
    angle = np.zeros(2, dtype=np.float32)
    ang_vel = np.zeros(2, dtype=np.float32)
    n = 2
    
    mats = np.array([[[-1.0]]], dtype=np.float32) 
    params = np.array([[0.0, 0.1, 1.0, 0.0, 3.0]], dtype=np.float32) # Type 3 = Repel Only (updated Enum order in mind? No, types.py says 3)
    
    species_params = np.array([[0.05, 1.0, 0.0]], dtype=np.float32)
    
    cell_size = 0.5
    grid_counts = np.zeros((4,4), dtype=np.int32)
    grid_cells = np.zeros((4,4,10), dtype=np.int32)
    # Fill
    grid_counts[2,2] = 2
    grid_cells[2,2,0] = 0
    grid_cells[2,2,1] = 1
    
    forces, torques = compute_forces_multi(
        pos, vel, colors, angle, ang_vel, n,
        mats, params, species_params,
        0.0, 10.0,
        grid_counts, grid_cells, cell_size,
        0.01, 1.0, 0.0
    )
    
    # Should repel (Negative Force on P0)
    assert forces[0, 0] < 0 
