"""
Numba-optimized physics kernels.
Multi-Rule Support.
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
    """
    grid_counts.fill(0)
    h, w = grid_counts.shape
    max_p = grid_cells.shape[2]
    
    for i in range(n_active):
        cx = int((pos[i, 0] + 1.0) / cell_size)
        cy = int((pos[i, 1] + 1.0) / cell_size)
        
        if cx < 0: cx = 0
        elif cx >= w: cx = w - 1
        if cy < 0: cy = 0
        elif cy >= h: cy = h - 1
        
        idx = grid_counts[cy, cx]
        if idx < max_p:
            grid_cells[cy, cx, idx] = i
            grid_counts[cy, cx] += 1

@njit(parallel=True, fastmath=True)
def compute_forces_multi(
    pos, colors, n_active,
    # Rule Arrays:
    # rule_matrices: (N_Rules, T, T)
    # rule_params: (N_Rules, 5) -> [min_r, max_r, strength, softening, force_type]
    rule_matrices, rule_params,
    grid_counts, grid_cells, cell_size,
    dt, friction, gravity
):
    """
    Compute interactions using multiple force rules.
    """
    forces = np.zeros_like(pos)
    n_rules = rule_matrices.shape[0]
    
    # Grid bounds
    grid_h, grid_w = grid_counts.shape
    
    # Iterate Particles
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
                    
                    # Vector to other
                    other_pos = pos[j]
                    dx_vec = other_pos[0] - p_pos[0]
                    dy_vec = other_pos[1] - p_pos[1]
                    
                    # Pre-calc squared distance
                    dist_sq = dx_vec*dx_vec + dy_vec*dy_vec
                    if dist_sq < 1e-8: dist_sq = 1e-8
                    
                    dist = np.sqrt(dist_sq)
                    
                    # Normalized direction
                    nx_vec = dx_vec / dist
                    ny_vec = dy_vec / dist
                    
                    # Iterate Rules
                    for r in range(n_rules):
                        min_r = rule_params[r, 0]
                        max_r = rule_params[r, 1]
                        strength = rule_params[r, 2]
                        softening = rule_params[r, 3]
                        ftype = int(rule_params[r, 4])
                        
                        # Distance Check
                        # Optimization: If dist > max_r, skip
                        if dist >= max_r:
                            continue
                            
                        # Lookup Factor
                        factor = rule_matrices[r, p_type, colors[j]]
                        if factor == 0.0:
                            continue
                            
                        force_val = 0.0
                        
                        # --- LINEAR (Particle Life) ---
                        if ftype == 0:
                            if dist < min_r:
                                # Universal Repulsion inside min_r
                                # F = (dist/min_r - 1)
                                force_val = (dist / min_r) - 1.0
                                force_val *= 2.0 # Hard Repulsion Multiplier
                            else:
                                # Linear Attraction/Repulsion Peak
                                range_len = max_r - min_r
                                numer = np.abs(2.0 * dist - range_len - 2.0 * min_r)
                                peak = 1.0 - (numer / range_len)
                                force_val = factor * peak
                        
                        # --- INVERSE SQUARE (Gravity) ---
                        elif ftype == 1:
                            # F = G * m1 * m2 / r^2
                            # Here "m1*m2" is our matrix factor.
                            # Use softening: 1 / (r + eps)^2 or r / (r^2 + eps^2)^1.5 (Spline softening)
                            # Simple Plummet: 1 / (max(dist, softening))^2
                            # We want it to go to 0 at max_r?
                            # Usually gravity is infinite range, but we employ cutoffs for performance.
                            # Let's just do standard inverse square with cutoff.
                            
                            denom = dist + softening
                            force_val = factor / (denom * denom)
                            
                            # Fade out at boundary to avoid pops?
                            # Simple linear fade: (1 - dist/max_r)
                            fade = 1.0 - (dist / max_r)
                            force_val *= fade
                            
                        # --- REPEL ONLY ---
                        elif ftype == 2:
                            if dist < max_r:
                                # Like Linear repulsion but across whole range
                                # factor is usually negative for repulsion
                                force_val = factor * (1.0 - (dist / max_r))
                        
                        # Apply Strength Scaling
                        fx += nx_vec * force_val * strength
                        fy += ny_vec * force_val * strength
                        
        forces[i, 0] = fx
        forces[i, 1] = fy - gravity
        
    return forces

@njit(parallel=True, fastmath=True)
def integrate(pos, vel, forces, n_active, dt, friction, bounds):
    """
    Euler integration with boundary handling.
    """
    lower = bounds[0]
    upper = bounds[1]
    damp = -0.7 
    
    for i in prange(n_active):
        vel[i, 0] = (vel[i, 0] + forces[i, 0] * dt) * friction
        vel[i, 1] = (vel[i, 1] + forces[i, 1] * dt) * friction
        
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
