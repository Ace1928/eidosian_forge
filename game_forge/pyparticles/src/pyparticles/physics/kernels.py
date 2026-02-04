"""
Numba-optimized physics kernels.
Separated from state management for pure compilation speed.
"""
import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def get_cell_coords(pos, cell_size, grid_width):
    """Map position to grid indices."""
    cx = int((pos[0] + 1.0) / cell_size)
    cy = int((pos[1] + 1.0) / cell_size)
    return cx, cy

@njit(fastmath=True)
def fill_grid(pos, n_active, cell_size, grid_counts, grid_cells):
    """
    Populate spatial partition grid.
    pos: (N, 2)
    n_active: int, number of particles to process
    grid_counts: (H, W)
    grid_cells: (H, W, MaxP)
    """
    # Clear counts
    grid_counts.fill(0)
    
    h, w = grid_counts.shape
    max_p = grid_cells.shape[2]
    
    for i in range(n_active):
        cx = int((pos[i, 0] + 1.0) / cell_size)
        cy = int((pos[i, 1] + 1.0) / cell_size)
        
        # Clamp
        if cx < 0: cx = 0
        elif cx >= w: cx = w - 1
        
        if cy < 0: cy = 0
        elif cy >= h: cy = h - 1
        
        # Insert
        idx = grid_counts[cy, cx]
        if idx < max_p:
            grid_cells[cy, cx, idx] = i
            grid_counts[cy, cx] += 1

@njit(parallel=True, fastmath=True)
def compute_forces(
    pos, colors, n_active, matrix,
    grid_counts, grid_cells, cell_size,
    dt, friction, max_r, min_r, rep_strength, gravity
):
    """
    Compute interactions and output forces.
    """
    forces = np.zeros_like(pos)
    
    # Pre-calc square radii
    max_r_sq = max_r * max_r
    
    # Grid bounds
    grid_h, grid_w = grid_counts.shape
    
    for i in prange(n_active):
        p_pos = pos[i]
        p_type = colors[i]
        fx = 0.0
        fy = 0.0
        
        # Get cell
        cx = int((p_pos[0] + 1.0) / cell_size)
        cy = int((p_pos[1] + 1.0) / cell_size)
        
        # Neighbor search
        for dy in range(-1, 2):
            ny = cy + dy
            if ny < 0 or ny >= grid_h: continue
            
            for dx in range(-1, 2):
                nx = cx + dx
                if nx < 0 or nx >= grid_w: continue
                
                count = grid_counts[ny, nx]
                for k in range(count):
                    j = grid_cells[ny, nx, k]
                    
                    if i == j: continue
                    
                    # Interaction
                    other_pos = pos[j]
                    dx_vec = other_pos[0] - p_pos[0]
                    dy_vec = other_pos[1] - p_pos[1]
                    
                    # Fast distance check
                    if abs(dx_vec) > max_r or abs(dy_vec) > max_r:
                        continue
                        
                    dist_sq = dx_vec*dx_vec + dy_vec*dy_vec
                    
                    if dist_sq >= max_r_sq or dist_sq < 1e-8:
                        continue
                    
                    dist = np.sqrt(dist_sq)
                    
                    # Normalized direction
                    nx_vec = dx_vec / dist
                    ny_vec = dy_vec / dist
                    
                    force_mag = 0.0
                    
                    # Repulsion
                    if dist < min_r:
                        force_mag = rep_strength * ((dist / min_r) - 1.0)
                    else:
                        # Attraction
                        # Standard Particle Life Formula
                        range_len = max_r - min_r
                        numer = np.abs(2.0 * dist - range_len - 2.0 * min_r)
                        peak = 1.0 - (numer / range_len)
                        
                        factor = matrix[p_type, colors[j]]
                        force_mag = factor * peak
                        
                    fx += nx_vec * force_mag
                    fy += ny_vec * force_mag
                    
        forces[i, 0] = fx
        forces[i, 1] = fy - gravity # Apply gravity here
        
    return forces

@njit(parallel=True, fastmath=True)
def integrate(pos, vel, forces, n_active, dt, friction, bounds):
    """
    Euler integration with boundary handling.
    """
    lower = bounds[0]
    upper = bounds[1]
    damp = -0.7 # Wall bounce coefficient
    
    for i in prange(n_active):
        # Symplectic Euler (sort of) - update vel then pos? 
        # Or standard Euler: pos += vel * dt.
        # Original used: vel = (vel + F*dt) * friction
        
        # Update Velocity
        vel[i, 0] = (vel[i, 0] + forces[i, 0] * dt) * friction
        vel[i, 1] = (vel[i, 1] + forces[i, 1] * dt) * friction
        
        # Update Position
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        
        # Walls
        if pos[i, 0] < lower:
            pos[i, 0] = lower
            vel[i, 0] *= damp
        elif pos[i, 0] > upper:
            pos[i, 0] = upper
            vel[i, 0] *= damp
            
        if pos[i, 1] < lower:
            pos[i, 1] = lower
            vel[i, 1] *= damp
        elif pos[i, 1] > upper:
            pos[i, 1] = upper
            vel[i, 1] *= damp
