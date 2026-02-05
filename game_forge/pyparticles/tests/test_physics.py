import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pyparticles.core.types import SimulationConfig
from pyparticles.physics.engine import PhysicsEngine
from pyparticles.physics.kernels import get_cell_coords, fill_grid, compute_forces_multi, integrate

def test_engine_rules():
    cfg = SimulationConfig.default()
    engine = PhysicsEngine(cfg)
    assert len(engine.rules) == 2 # Default: Linear + Gravity
    
    # Check packing
    mats, params = engine._pack_rules()
    assert mats.shape == (2, 6, 6)
    assert params.shape == (2, 5)

def test_kernel_compute_forces_multi_logic():
    # Setup 2 particles
    pos = np.array([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32)
    colors = np.array([0, 0], dtype=np.int32)
    n = 2
    
    # Setup Rules
    # Rule 0: Linear Repulsion (dist < 0.1)
    mats = np.zeros((2, 1, 1), dtype=np.float32)
    mats[0, 0, 0] = 1.0 # Attraction factor
    mats[1, 0, 0] = 0.5 # Gravity factor
    
    params = np.zeros((2, 5), dtype=np.float32)
    # R0: Linear, min 0.02, max 0.1
    params[0, 0] = 0.02
    params[0, 1] = 0.1
    params[0, 2] = 1.0 # Strength
    params[0, 3] = 0.0 # Softening
    params[0, 4] = 0.0 # Type Linear
    
    # R1: Gravity, min 0.01, max 0.5
    params[1, 0] = 0.01
    params[1, 1] = 0.5
    params[1, 2] = 1.0
    params[1, 3] = 0.05
    params[1, 4] = 1.0 # Type InvSq
    
    # Grid
    cell_size = 0.5
    grid_counts = np.zeros((4, 4), dtype=np.int32)
    grid_cells = np.zeros((4, 4, 10), dtype=np.int32)
    fill_grid(pos, n, cell_size, grid_counts, grid_cells)
    
    dt = 0.01
    fric = 1.0
    grav = 0.0
    
    forces = compute_forces_multi(
        pos, colors, n, mats, params,
        grid_counts, grid_cells, cell_size,
        dt, fric, grav
    )
    
    # Analyze Forces
    # Dist = 0.05.
    # Linear: 0.02 < 0.05 < 0.1 -> Attraction.
    # Gravity: 0.05 < 0.5 -> Attraction.
    # Total X force on P0 should be Positive (pulled right)
    
    assert forces[0, 0] > 0
    assert forces[1, 0] < 0

def test_kernel_repel_only():
    # Test Type 2
    pos = np.array([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32)
    colors = np.array([0, 0], dtype=np.int32)
    n = 2
    
    mats = np.array([[[-1.0]]], dtype=np.float32) # Negative factor
    params = np.array([[0.0, 0.1, 1.0, 0.0, 2.0]], dtype=np.float32) # Type 2
    
    cell_size = 0.5
    grid_counts = np.zeros((4,4), dtype=int)
    grid_cells = np.zeros((4,4,10), dtype=int)
    fill_grid(pos, n, cell_size, grid_counts, grid_cells)
    
    forces = compute_forces_multi(pos, colors, n, mats, params, grid_counts, grid_cells, cell_size, 0.01, 1.0, 0.0)
    
    # Should repel
    assert forces[0, 0] < 0