"""
Physics Kernels.
Updated for Velocity Verlet Integration & Thermostat.
"""
import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def fill_grid(pos, n_active, cell_size, grid_counts, grid_cells):
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
    pos, colors, angle, n_active,
    # Rules
    rule_matrices, rule_params,
    # Species: (T, 3) -> [rad, freq, amp]
    species_params, 
    wave_strength, wave_exp,
    # Grid
    grid_counts, grid_cells, cell_size,
    gravity
):
    """
    Computes forces and torques based on current positions.
    Pure function: Input Pos -> Output Force/Torque.
    """
    forces = np.zeros_like(pos)
    torques = np.zeros_like(angle)
    
    n_rules = rule_matrices.shape[0]
    grid_h, grid_w = grid_counts.shape
    
    for i in prange(n_active):
        p_pos = pos[i]
        p_type = colors[i]
        p_angle = angle[i]
        
        p_rad = species_params[p_type, 0]
        p_freq = species_params[p_type, 1]
        p_amp = species_params[p_type, 2]
        
        fx = 0.0
        fy = 0.0
        tau = 0.0
        
        cx = int((p_pos[0] + 1.0) / cell_size)
        cy = int((p_pos[1] + 1.0) / cell_size)
        
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
                    
                    other_pos = pos[j]
                    dx_vec = other_pos[0] - p_pos[0]
                    dy_vec = other_pos[1] - p_pos[1]
                    dist_sq = dx_vec*dx_vec + dy_vec*dy_vec
                    if dist_sq < 1e-8: dist_sq = 1e-8
                    dist = np.sqrt(dist_sq)
                    
                    nx_vec = dx_vec / dist
                    ny_vec = dy_vec / dist
                    
                    # --- Long Range ---
                    for r in range(n_rules):
                        max_r = rule_params[r, 1]
                        if max_r == 0 or dist >= max_r: continue
                        
                        factor = rule_matrices[r, p_type, colors[j]]
                        if factor == 0.0: continue
                        
                        min_r = rule_params[r, 0]
                        strength = rule_params[r, 2]
                        softening = rule_params[r, 3]
                        ftype = int(rule_params[r, 4])
                        
                        force_val = 0.0
                        if ftype == 0: # Linear
                            if dist < min_r:
                                force_val = (dist / min_r) - 1.0
                                force_val *= 3.0 # Stronger Repulsion
                            else:
                                range_len = max_r - min_r
                                numer = np.abs(2.0 * dist - range_len - 2.0 * min_r)
                                peak = 1.0 - (numer / range_len)
                                force_val = factor * peak
                        elif ftype == 1: # InvSq
                            denom = dist + softening
                            force_val = factor / (denom * denom)
                            force_val *= (1.0 - (dist/max_r))
                        elif ftype == 2: # InvCube
                            denom = dist + softening
                            force_val = factor / (denom * denom * denom)
                            force_val *= (1.0 - (dist/max_r))
                        elif ftype == 3: # Repel
                            force_val = factor * (1.0 - (dist / max_r))
                            
                        fx += nx_vec * force_val * strength
                        fy += ny_vec * force_val * strength

                    # --- Wave Mechanics ---
                    interaction_range = p_rad + p_amp + 0.1
                    
                    if dist < interaction_range:
                        o_type = colors[j]
                        o_rad = species_params[o_type, 0]
                        o_freq = species_params[o_type, 1]
                        o_amp = species_params[o_type, 2]
                        o_angle = angle[j]
                        
                        phi_ij = np.arctan2(dy_vec, dx_vec)
                        theta_i = phi_ij - p_angle
                        theta_j = (phi_ij + np.pi) - o_angle
                        
                        h_i = p_amp * np.cos(p_freq * theta_i)
                        h_j = o_amp * np.cos(o_freq * theta_j)
                        
                        r_i = p_rad + h_i
                        r_j = o_rad + h_j
                        
                        gap = dist - (r_i + r_j)
                        
                        if gap < 0:
                            M = 1.0
                            if p_amp > 0 and o_amp > 0:
                                norm_i = h_i / p_amp
                                norm_j = h_j / o_amp
                                M = 1.0 + 0.5 * (norm_i * norm_j)
                            
                            clamped_gap = max(gap, -0.5) 
                            f_wave = wave_strength * M * (np.exp(-clamped_gap * wave_exp) - 1.0)
                            
                            fx -= nx_vec * f_wave
                            fy -= ny_vec * f_wave
                            
                            slope_i = 0.0
                            if r_i > 0.001:
                                slope_i = (-p_amp * p_freq * np.sin(p_freq * theta_i)) / r_i
                            
                            torque_i = f_wave * slope_i * r_i
                            tau += torque_i * 0.5
        
        forces[i, 0] = fx
        forces[i, 1] = fy - gravity
        torques[i] = tau
        
    return forces, torques

@njit(parallel=True, fastmath=True)
def apply_thermostat(vel, n_active, target_temp, coupling, dt):
    """
    Berendsen Thermostat.
    Scales velocities to approach target temperature.
    """
    ke_sum = 0.0
    for i in prange(n_active):
        vx = vel[i, 0]
        vy = vel[i, 1]
        ke_sum += 0.5 * (vx*vx + vy*vy)
        
    if n_active > 0:
        ke_avg = ke_sum / n_active
    else:
        ke_avg = 0.0
        
    if ke_avg < 1e-6: ke_avg = 1e-6
    
    ratio = target_temp / ke_avg
    # Avoid sqrt of negative if ratio very small (unlikely but robust)
    if ratio < 0: ratio = 0
    scale = np.sqrt(1.0 + coupling * (ratio - 1.0))
    
    # Soft Limit
    if scale > 1.05: scale = 1.05
    if scale < 0.95: scale = 0.95
    
    for i in prange(n_active):
        vel[i, 0] *= scale
        vel[i, 1] *= scale

@njit(parallel=True, fastmath=True)
def integrate_verlet_1(pos, vel, angle, ang_vel, forces, torques, n_active, dt, bounds):
    """
    First half of Velocity Verlet.
    v(t + 0.5dt) = v(t) + 0.5 * a(t) * dt
    r(t + dt) = r(t) + v(t + 0.5dt) * dt
    """
    lower = bounds[0]
    upper = bounds[1]
    
    for i in prange(n_active):
        # Half step velocity
        vel[i, 0] += 0.5 * forces[i, 0] * dt
        vel[i, 1] += 0.5 * forces[i, 1] * dt
        
        # Full step position
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        
        # Walls (Simple reflection)
        if pos[i, 0] < lower:
            pos[i, 0] = lower
            vel[i, 0] *= -1.0 # Elastic
        elif pos[i, 0] > upper:
            pos[i, 0] = upper
            vel[i, 0] *= -1.0
            
        if pos[i, 1] < lower:
            pos[i, 1] = lower
            vel[i, 1] *= -1.0
        elif pos[i, 1] > upper:
            pos[i, 1] = upper
            vel[i, 1] *= -1.0
            
        # Angular (Simple Euler for now as Torque depends on F, and F depends on Pos)
        # For strict Verlet, we need Torque(t) and Torque(t+dt).
        # We can treat angular dynamics as Euler-Cromer or Verlet if needed.
        # But torque is derived from F_wave which is pos dependent.
        # So we can do Verlet for Angle too.
        
        ang_vel[i] += 0.5 * torques[i] * dt
        angle[i] += ang_vel[i] * dt

@njit(parallel=True, fastmath=True)
def integrate_verlet_2(vel, ang_vel, forces, torques, n_active, dt, friction):
    """
    Second half of Velocity Verlet.
    v(t + dt) = v(t + 0.5dt) + 0.5 * a(t + dt) * dt
    """
    for i in prange(n_active):
        vel[i, 0] += 0.5 * forces[i, 0] * dt
        vel[i, 1] += 0.5 * forces[i, 1] * dt
        
        ang_vel[i] += 0.5 * torques[i] * dt
        
        # Apply Drag/Friction at end of step?
        if friction < 1.0:
            vel[i, 0] *= (1.0 - friction * dt)
            vel[i, 1] *= (1.0 - friction * dt)
            ang_vel[i] *= 0.98 # Rot drag