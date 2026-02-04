import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pyparticles.core.types import SimulationConfig
from pyparticles.physics.engine import PhysicsEngine
from pyparticles.physics.kernels import get_cell_coords, fill_grid, compute_forces, integrate

def test_engine_init():
    cfg = SimulationConfig.default()
    cfg.num_particles = 100
    engine = PhysicsEngine(cfg)
    assert engine.state.active == 100
    assert engine.matrix.shape == (6, 6)

def test_engine_reset():
    cfg = SimulationConfig.default()
    engine = PhysicsEngine(cfg)
    engine.state.pos[0] = [100.0, 100.0]
    engine.reset()
    assert engine.state.pos[0, 0] <= 1.0

def test_engine_set_active():
    cfg = SimulationConfig.default()
    engine = PhysicsEngine(cfg)
    engine.set_active_count(200)
    assert engine.state.active == 200
    engine.set_active_count(cfg.max_particles + 1000)
    assert engine.state.active == cfg.max_particles

def test_engine_update():
    cfg = SimulationConfig.default()
    cfg.num_particles = 10
    engine = PhysicsEngine(cfg)
    engine.update(0.01)

def test_kernel_get_cell_coords():
    c = get_cell_coords(np.array([-1.0, -1.0]), 0.1, 10)
    assert c == (0, 0)
    c2 = get_cell_coords(np.array([1.0, 1.0]), 0.1, 10)
    # 2.0 / 0.1 = 20
    assert c2 == (20, 20)

def test_kernel_fill_grid():
    pos = np.array([[-0.95, -0.95], [0.95, 0.95]], dtype=np.float32)
    n = 2
    cell_size = 0.5 
    grid_counts = np.zeros((4, 4), dtype=np.int32)
    grid_cells = np.zeros((4, 4, 10), dtype=np.int32)
    fill_grid(pos, n, cell_size, grid_counts, grid_cells)
    assert grid_counts[0, 0] == 1 
    assert grid_counts[3, 3] == 1 
    
    # Test boundary clamping for grid fill
    pos_out = np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32)
    grid_counts.fill(0)
    fill_grid(pos_out, 2, cell_size, grid_counts, grid_cells)
    assert grid_counts[0, 0] == 1
    assert grid_counts[3, 3] == 1

def test_kernel_compute_forces_interactions():
    # Test forces between two close particles
    pos = np.array([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32) # dist 0.05
    colors = np.array([0, 0], dtype=np.int32)
    n = 2
    matrix = np.ones((1, 1), dtype=np.float32) # Attraction 1.0
    
    # Grid setup
    cell_size = 0.2
    grid_counts = np.zeros((10, 10), dtype=np.int32)
    grid_cells = np.zeros((10, 10, 10), dtype=np.int32)
    fill_grid(pos, n, cell_size, grid_counts, grid_cells)
    
    dt = 0.01
    friction = 1.0
    max_r = 0.1
    min_r = 0.02 # Repulsion radius
    rep_s = 2.0
    grav = 0.0
    
    # Case 1: Attraction
    forces = compute_forces(pos, colors, n, matrix, grid_counts, grid_cells, cell_size, dt, friction, max_r, min_r, rep_s, grav)
    assert forces[0, 0] > 0
    assert forces[1, 0] < 0
    
    # Case 2: Repulsion
    pos2 = np.array([[0.0, 0.0], [0.01, 0.0]], dtype=np.float32)
    fill_grid(pos2, n, cell_size, grid_counts, grid_cells)
    forces2 = compute_forces(pos2, colors, n, matrix, grid_counts, grid_cells, cell_size, dt, friction, max_r, min_r, rep_s, grav)
    assert forces2[0, 0] < 0
    assert forces2[1, 0] > 0

    # Case 3: Gravity
    grav = 1.0
    forces3 = compute_forces(pos, colors, n, matrix, grid_counts, grid_cells, cell_size, dt, friction, max_r, min_r, rep_s, grav)
    assert forces3[0, 1] < 0 # Gravity pulls down (-Y)

def test_kernel_integration_bounds():
    # Test integration and wall collision
    # X Axis Upper
    pos = np.array([[0.99, 0.0]], dtype=np.float32)
    vel = np.array([[1.0, 0.0]], dtype=np.float32)
    forces = np.zeros_like(pos)
    n = 1
    dt = 0.1
    friction = 1.0
    bounds = np.array([-1.0, 1.0], dtype=np.float32)
    
    integrate(pos, vel, forces, n, dt, friction, bounds)
    assert pos[0, 0] == 1.0
    assert vel[0, 0] < 0
    
    # X Axis Lower
    pos[0] = [-0.99, 0.0]
    vel[0] = [-1.0, 0.0]
    integrate(pos, vel, forces, n, dt, friction, bounds)
    assert pos[0, 0] == -1.0
    assert vel[0, 0] > 0
    
    # Y Axis Upper
    pos[0] = [0.0, 0.99]
    vel[0] = [0.0, 1.0]
    integrate(pos, vel, forces, n, dt, friction, bounds)
    assert pos[0, 1] == 1.0
    assert vel[0, 1] < 0

    # Y Axis Lower
    pos[0] = [0.0, -0.99]
    vel[0] = [0.0, -1.0]
    integrate(pos, vel, forces, n, dt, friction, bounds)
    assert pos[0, 1] == -1.0
    assert vel[0, 1] > 0
