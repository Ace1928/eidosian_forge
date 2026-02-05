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
    colors = np.array([0, 0], dtype=np.int32)
    angle = np.zeros(2, dtype=np.float32)
    n = 2
    
    # Setup Rules
    mats = np.zeros((2, 1, 1), dtype=np.float32)
    mats[0, 0, 0] = 1.0 
    mats[1, 0, 0] = 0.5 
    
    params = np.zeros((2, 5), dtype=np.float32)
    # R0: Linear
    params[0, 0] = 0.02
    params[0, 1] = 0.1
    params[0, 2] = 1.0
    params[0, 3] = 0.0
    params[0, 4] = 0.0 # ForceType.LINEAR
    
    # R1: InvSq
    params[1, 0] = 0.01
    params[1, 1] = 0.5
    params[1, 2] = 1.0
    params[1, 3] = 0.05
    params[1, 4] = 1.0 # ForceType.INVERSE_SQUARE
    
    # Species Params (1 Type)
    # Rad, Freq, Amp
    species_params = np.array([[0.05, 1.0, 0.0]], dtype=np.float32)
    
    # Grid
    cell_size = 0.5
    grid_counts = np.zeros((4, 4), dtype=np.int32)
    grid_cells = np.zeros((4, 4, 10), dtype=np.int32)
    cx0, cy0 = 2, 2 
    grid_counts[cy0, cx0] = 2
    grid_cells[cy0, cx0, 0] = 0
    grid_cells[cy0, cx0, 1] = 1
    
    grav = 0.0
    wave_str = 0.0 
    wave_exp = 10.0

    forces, torques = compute_forces_multi(
        pos, colors, angle, n,
        mats, params, species_params,
        wave_str, wave_exp,
        grid_counts, grid_cells, cell_size,
        grav
    )

    # Code says: fx += nx_vec * force_val * strength.
    # nx_vec for P0 is (P1-P0) = (1,0).
    # Linear Rule at d=0.05 is attractive (>0)
    assert forces[0, 0] > 0.0
    assert forces[1, 0] < 0.0 

def test_kernel_repel_only():
    pos = np.array([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32)
    colors = np.array([0, 0], dtype=np.int32)
    angle = np.zeros(2, dtype=np.float32)
    n = 2
    
    mats = np.array([[[-1.0]]], dtype=np.float32) 
    params = np.array([[0.0, 0.1, 1.0, 0.0, 3.0]], dtype=np.float32) # Type 3 = Repel Only
    
    species_params = np.array([[0.05, 1.0, 0.0]], dtype=np.float32)
    
    cell_size = 0.5
    grid_counts = np.zeros((4,4), dtype=np.int32)
    grid_cells = np.zeros((4,4,10), dtype=np.int32)
    grid_counts[2,2] = 2
    grid_cells[2,2,0] = 0
    grid_cells[2,2,1] = 1

    forces, torques = compute_forces_multi(
        pos, colors, angle, n,
        mats, params, species_params,
        0.0, 10.0,
        grid_counts, grid_cells, cell_size,
        0.0
    )

    # Repel Only: factor=-1.0. fx should be negative for P0
    assert forces[0, 0] < 0.0
    assert forces[1, 0] > 0.0