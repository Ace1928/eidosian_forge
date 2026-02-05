import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pyparticles.core.types import SimulationConfig
from pyparticles.physics.engine import PhysicsEngine
from pyparticles.physics.kernels import compute_forces_multi

def test_engine_rules():
    cfg = SimulationConfig.default()
    engine = PhysicsEngine(cfg)
    assert len(engine.rules) == 3  # Linear, Gravity, Core Repulsion
    
    # Check packing - matrices match num_types from config
    mats, params = engine._pack_rules()
    assert mats.shape == (3, cfg.num_types, cfg.num_types)
    assert params.shape == (3, 8)  # V6: Extended params format

def test_kernel_compute_forces_multi_logic():
    # Setup 2 particles
    pos = np.array([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32)
    colors = np.array([0, 0], dtype=np.int32)
    angle = np.zeros(2, dtype=np.float32)
    n = 2
    
    # Setup Rules - V6: 8-column params
    mats = np.zeros((2, 1, 1), dtype=np.float32)
    mats[0, 0, 0] = 1.0 
    mats[1, 0, 0] = 0.5 
    
    params = np.zeros((2, 8), dtype=np.float32)
    # R0: Linear
    params[0, 0] = 0.02   # min_r
    params[0, 1] = 0.1    # max_r
    params[0, 2] = 1.0    # strength
    params[0, 3] = 0.0    # softening
    params[0, 4] = 0.0    # ForceType.LINEAR
    
    # R1: InvSq
    params[1, 0] = 0.01   # min_r
    params[1, 1] = 0.5    # max_r
    params[1, 2] = 1.0    # strength
    params[1, 3] = 0.05   # softening
    params[1, 4] = 1.0    # ForceType.INVERSE_SQUARE
    
    # Species Params (1 Type) - V6: 6 columns [rad, freq, amp, inertia, spin_fric, base_spin]
    species_params = np.array([[0.05, 1.0, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    
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
        grav, 1.0  # half_world
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
    params = np.array([[0.0, 0.1, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # V6: 8 cols
    
    # V6: 6-column species params
    species_params = np.array([[0.05, 1.0, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    
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
        0.0, 1.0  # gravity, half_world
    )

    # Repel Only: factor=-1.0. fx should be negative for P0
    assert forces[0, 0] < 0.0
    assert forces[1, 0] > 0.0