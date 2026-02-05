"""
Eidosian PyParticles V6 - Physics Kernels

Numba JIT-compiled kernels for high-performance particle simulation.
Supports: Velocity Verlet integration, multiple force types, wave mechanics,
          Berendsen thermostat, and spatial hashing.
"""

import numpy as np
from numba import njit, prange

# ============================================================================
# SPATIAL HASHING
# ============================================================================

@njit(fastmath=True, cache=True)
def fill_grid(pos, n_active, cell_size, grid_counts, grid_cells):
    """
    Build spatial hash grid for O(N) neighbor queries.
    
    Each particle is assigned to a cell based on its position.
    Grid coordinates are in [0, grid_size) corresponding to world [-1, 1].
    """
    grid_counts.fill(0)
    h, w = grid_counts.shape
    max_p = grid_cells.shape[2]
    
    for i in range(n_active):
        # Map world [-1,1] to grid [0, w) and [0, h)
        cx = int((pos[i, 0] + 1.0) / cell_size)
        cy = int((pos[i, 1] + 1.0) / cell_size)
        
        # Clamp to grid bounds
        if cx < 0: cx = 0
        elif cx >= w: cx = w - 1
        if cy < 0: cy = 0
        elif cy >= h: cy = h - 1
        
        idx = grid_counts[cy, cx]
        if idx < max_p:
            grid_cells[cy, cx, idx] = i
            grid_counts[cy, cx] += 1

# ============================================================================
# FORCE CALCULATIONS
# ============================================================================

@njit(fastmath=True, cache=True, inline='always')
def compute_force_by_type(
    dist: float,
    min_r: float,
    max_r: float,
    factor: float,
    strength: float,
    ftype: int,
    softening: float,
    param1: float,
    param2: float,
    param3: float,
) -> float:
    """
    Unified force calculation for all force types.
    
    Force types:
        0: LINEAR (particle life)
        1: INVERSE_SQUARE (gravity/Coulomb)
        2: INVERSE_CUBE (dipole)
        3: REPEL_ONLY (pure repulsion)
        4: INVERSE (1/r)
        5: YUKAWA (screened Coulomb)
        6: LENNARD_JONES (molecular)
        7: MORSE (bond-like)
        8: GAUSSIAN (soft localized)
        9: EXPONENTIAL (decay)
    """
    if max_r > 0 and dist >= max_r:
        return 0.0
    
    force_val = 0.0
    
    if ftype == 0:  # LINEAR
        if dist < min_r:
            # Strong repulsion inside core
            force_val = (dist / min_r) - 1.0
            force_val *= 3.0
        else:
            # Bell curve attraction/repulsion
            range_len = max_r - min_r
            if range_len > 0:
                numer = np.abs(2.0 * dist - range_len - 2.0 * min_r)
                peak = 1.0 - (numer / range_len)
                force_val = factor * peak
                
    elif ftype == 1:  # INVERSE_SQUARE
        denom = dist + softening
        force_val = factor / (denom * denom)
        if max_r > 0:
            force_val *= (1.0 - dist / max_r)
            
    elif ftype == 2:  # INVERSE_CUBE
        denom = dist + softening
        force_val = factor / (denom * denom * denom)
        if max_r > 0:
            force_val *= (1.0 - dist / max_r)
            
    elif ftype == 3:  # REPEL_ONLY
        force_val = factor * (1.0 - dist / max_r) if max_r > 0 else factor
        
    elif ftype == 4:  # INVERSE
        denom = dist + softening
        force_val = factor / denom
        if max_r > 0:
            force_val *= (1.0 - dist / max_r)
            
    elif ftype == 5:  # YUKAWA
        # F = k * (1/r² + 1/(λr)) * exp(-r/λ)
        decay_length = param1 if param1 > 0 else 0.1
        r = dist + softening
        exp_term = np.exp(-dist / decay_length)
        inv_r = 1.0 / r
        force_val = factor * (inv_r * inv_r + inv_r / decay_length) * exp_term
        
    elif ftype == 6:  # LENNARD_JONES
        # F = 24ε/r * [2(σ/r)¹² - (σ/r)⁶]
        sigma = param1 if param1 > 0 else 0.05
        r = dist + softening
        ratio = sigma / r
        r6 = ratio ** 6
        r12 = r6 * r6
        force_val = (24.0 * factor / r) * (2.0 * r12 - r6)
        
    elif ftype == 7:  # MORSE
        # F = 2aD * [exp(-a(r-r₀)) - exp(-2a(r-r₀))]
        r0 = param1 if param1 > 0 else 0.1
        well_width = param2 if param2 > 0 else 5.0
        dr = dist - r0
        exp1 = np.exp(-well_width * dr)
        exp2 = np.exp(-2.0 * well_width * dr)
        force_val = 2.0 * well_width * factor * (exp1 - exp2)
        
    elif ftype == 8:  # GAUSSIAN
        sigma = param1 if param1 > 0 else 0.1
        sigma_sq = sigma * sigma
        exp_term = np.exp(-dist * dist / (2.0 * sigma_sq))
        force_val = factor * (dist / sigma_sq) * exp_term
        
    elif ftype == 9:  # EXPONENTIAL
        decay_length = param1 if param1 > 0 else 0.1
        force_val = factor * np.exp(-dist / decay_length)
    
    return force_val * strength


@njit(parallel=True, fastmath=True, cache=True)
def compute_forces_multi(
    pos, colors, angle, n_active,
    # Rules: matrices (N_rules, T, T), params (N_rules, 8)
    rule_matrices, rule_params,
    # Species: (T, 3) -> [rad, freq, amp]
    species_params, 
    wave_strength, wave_exp,
    # Grid
    grid_counts, grid_cells, cell_size,
    gravity
):
    """
    Compute all forces and torques for active particles.
    
    This is the main physics kernel - called twice per Verlet step.
    
    Args:
        pos: (N, 2) particle positions
        colors: (N,) particle type indices
        angle: (N,) particle rotation angles
        n_active: Number of active particles
        rule_matrices: (R, T, T) interaction matrices
        rule_params: (R, 8) [min_r, max_r, strength, softening, ftype, p1, p2, p3]
        species_params: (T, 3) [radius, wave_freq, wave_amp]
        wave_strength: Wave repulsion strength
        wave_exp: Wave repulsion exponent
        grid_counts: (H, W) cell particle counts
        grid_cells: (H, W, M) cell particle indices
        cell_size: Grid cell size
        gravity: Downward gravity (positive = down)
        
    Returns:
        forces: (N, 2) force vectors
        torques: (N,) torque values
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
        
        # Get grid cell for this particle
        cx = int((p_pos[0] + 1.0) / cell_size)
        cy = int((p_pos[1] + 1.0) / cell_size)
        
        # Check 3x3 neighborhood
        for dy in range(-1, 2):
            ny = cy + dy
            if ny < 0 or ny >= grid_h:
                continue
            for dx in range(-1, 2):
                nx = cx + dx
                if nx < 0 or nx >= grid_w:
                    continue
                
                count = grid_counts[ny, nx]
                for k in range(count):
                    j = grid_cells[ny, nx, k]
                    if i == j:
                        continue
                    
                    # Distance calculation
                    other_pos = pos[j]
                    dx_vec = other_pos[0] - p_pos[0]
                    dy_vec = other_pos[1] - p_pos[1]
                    dist_sq = dx_vec * dx_vec + dy_vec * dy_vec
                    
                    if dist_sq < 1e-10:
                        dist_sq = 1e-10
                    dist = np.sqrt(dist_sq)
                    
                    # Unit vector toward other particle
                    nx_vec = dx_vec / dist
                    ny_vec = dy_vec / dist
                    
                    # ========== FORCE RULES ==========
                    for r in range(n_rules):
                        max_r = rule_params[r, 1]
                        if max_r == 0 or dist >= max_r:
                            continue
                        
                        factor = rule_matrices[r, p_type, colors[j]]
                        if factor == 0.0:
                            continue
                        
                        min_r = rule_params[r, 0]
                        strength = rule_params[r, 2]
                        softening = rule_params[r, 3]
                        ftype = int(rule_params[r, 4])
                        param1 = rule_params[r, 5]
                        param2 = rule_params[r, 6]
                        param3 = rule_params[r, 7]
                        
                        force_val = compute_force_by_type(
                            dist, min_r, max_r, factor, strength,
                            ftype, softening, param1, param2, param3
                        )
                        
                        fx += nx_vec * force_val
                        fy += ny_vec * force_val
                    
                    # ========== WAVE MECHANICS ==========
                    interaction_range = p_rad + abs(p_amp) + 0.1
                    
                    if dist < interaction_range and wave_strength > 0:
                        o_type = colors[j]
                        o_rad = species_params[o_type, 0]
                        o_freq = species_params[o_type, 1]
                        o_amp = species_params[o_type, 2]
                        o_angle = angle[j]
                        
                        # Angle from particle i to j
                        phi_ij = np.arctan2(dy_vec, dx_vec)
                        theta_i = phi_ij - p_angle
                        theta_j = (phi_ij + np.pi) - o_angle
                        
                        # Wave heights at contact point
                        h_i = p_amp * np.cos(p_freq * theta_i)
                        h_j = o_amp * np.cos(o_freq * theta_j)
                        
                        # Effective radii with wave deformation
                        r_i = p_rad + h_i
                        r_j = o_rad + h_j
                        
                        # Gap between effective surfaces
                        gap = dist - (r_i + r_j)
                        
                        if gap < 0:
                            # Interference multiplier
                            M = 1.0
                            if abs(p_amp) > 0.001 and abs(o_amp) > 0.001:
                                norm_i = h_i / p_amp
                                norm_j = h_j / o_amp
                                # Constructive: both peaks (+1) or both troughs (-1)
                                # Destructive: one peak, one trough
                                M = 1.0 + 0.5 * norm_i * norm_j
                            
                            # Exponential repulsion with clamped gap
                            clamped_gap = max(gap, -0.3)
                            f_wave = wave_strength * M * (np.exp(-clamped_gap * wave_exp) - 1.0)
                            
                            # Wave force is repulsive
                            fx -= nx_vec * f_wave
                            fy -= ny_vec * f_wave
                            
                            # Torque from wave slope
                            if abs(r_i) > 0.001:
                                slope_i = (-p_amp * p_freq * np.sin(p_freq * theta_i)) / r_i
                                torque_i = f_wave * slope_i * r_i
                                tau += torque_i * 0.5
        
        forces[i, 0] = fx
        forces[i, 1] = fy - gravity
        torques[i] = tau
    
    return forces, torques

# ============================================================================
# THERMOSTAT
# ============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def apply_thermostat(vel, n_active, target_temp, coupling, dt):
    """
    Berendsen thermostat for temperature control.
    
    Scales velocities toward target kinetic energy:
    λ = sqrt(1 + (dt/τ) * (T_target/T_current - 1))
    
    Applied AFTER velocity integration for correct NVT ensemble.
    """
    # Calculate current kinetic energy
    ke_sum = 0.0
    for i in prange(n_active):
        vx = vel[i, 0]
        vy = vel[i, 1]
        ke_sum += 0.5 * (vx * vx + vy * vy)
    
    if n_active > 0:
        ke_avg = ke_sum / n_active
    else:
        return
    
    if ke_avg < 1e-8:
        ke_avg = 1e-8
    
    # Berendsen scaling factor
    ratio = target_temp / ke_avg
    if ratio < 0:
        ratio = 0
    
    scale = np.sqrt(1.0 + coupling * (ratio - 1.0))
    
    # Soft limits to prevent instability
    if scale > 1.05:
        scale = 1.05
    if scale < 0.95:
        scale = 0.95
    
    # Scale all velocities
    for i in prange(n_active):
        vel[i, 0] *= scale
        vel[i, 1] *= scale

# ============================================================================
# INTEGRATION
# ============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def integrate_verlet_1(pos, vel, angle, ang_vel, forces, torques, n_active, dt, bounds):
    """
    First half of Velocity Verlet integration.
    
    v(t + 0.5dt) = v(t) + 0.5 * a(t) * dt
    r(t + dt) = r(t) + v(t + 0.5dt) * dt
    
    Also handles boundary reflection and angular motion.
    """
    lower = bounds[0]
    upper = bounds[1]
    
    for i in prange(n_active):
        # Half-step velocity update
        vel[i, 0] += 0.5 * forces[i, 0] * dt
        vel[i, 1] += 0.5 * forces[i, 1] * dt
        
        # Full-step position update
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        
        # Elastic wall reflection
        if pos[i, 0] < lower:
            pos[i, 0] = lower
            vel[i, 0] *= -1.0
        elif pos[i, 0] > upper:
            pos[i, 0] = upper
            vel[i, 0] *= -1.0
        
        if pos[i, 1] < lower:
            pos[i, 1] = lower
            vel[i, 1] *= -1.0
        elif pos[i, 1] > upper:
            pos[i, 1] = upper
            vel[i, 1] *= -1.0
        
        # Angular dynamics (half-step Verlet)
        ang_vel[i] += 0.5 * torques[i] * dt
        angle[i] += ang_vel[i] * dt


@njit(parallel=True, fastmath=True, cache=True)
def integrate_verlet_2(vel, ang_vel, forces, torques, n_active, dt, friction, angular_friction):
    """
    Second half of Velocity Verlet integration.
    
    v(t + dt) = v(t + 0.5dt) + 0.5 * a(t + dt) * dt
    
    Friction is applied as linear damping after velocity update.
    """
    for i in prange(n_active):
        # Complete velocity update
        vel[i, 0] += 0.5 * forces[i, 0] * dt
        vel[i, 1] += 0.5 * forces[i, 1] * dt
        ang_vel[i] += 0.5 * torques[i] * dt
        
        # Linear friction damping: v *= (1 - γdt)
        if friction > 0.0:
            damp = 1.0 - friction * dt
            if damp < 0.0:
                damp = 0.0
            vel[i, 0] *= damp
            vel[i, 1] *= damp
        
        # Angular friction damping
        if angular_friction > 0.0:
            ang_damp = 1.0 - angular_friction * dt
            if ang_damp < 0.0:
                ang_damp = 0.0
            ang_vel[i] *= ang_damp

# ============================================================================
# UTILITY KERNELS
# ============================================================================

@njit(fastmath=True, cache=True)
def compute_kinetic_energy(vel, n_active):
    """Calculate total and average kinetic energy."""
    total_ke = 0.0
    for i in range(n_active):
        vx = vel[i, 0]
        vy = vel[i, 1]
        total_ke += 0.5 * (vx * vx + vy * vy)
    
    avg_ke = total_ke / n_active if n_active > 0 else 0.0
    return total_ke, avg_ke


@njit(fastmath=True, cache=True)
def compute_momentum(vel, n_active):
    """Calculate total momentum (should be conserved without boundaries)."""
    px = 0.0
    py = 0.0
    for i in range(n_active):
        px += vel[i, 0]
        py += vel[i, 1]
    return px, py


@njit(fastmath=True, cache=True)
def compute_center_of_mass(pos, n_active):
    """Calculate center of mass position."""
    cx = 0.0
    cy = 0.0
    for i in range(n_active):
        cx += pos[i, 0]
        cy += pos[i, 1]
    
    if n_active > 0:
        cx /= n_active
        cy /= n_active
    
    return cx, cy