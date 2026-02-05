"""
Numba-optimized physics kernels.
Includes Inverse Cube and Wave Dynamics.
"""
import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def get_cell_coords(pos, cell_size, grid_width):
    cx = int((pos[0] + 1.0) / cell_size)
    cy = int((pos[1] + 1.0) / cell_size)
    return cx, cy

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
    pos, vel, colors, angle, ang_vel, n_active,
    # Rules
    rule_matrices, rule_params,
    # Wave Params: (T, 3) -> [radius, freq, amp]
    species_params, 
    wave_strength, wave_exp,
    # Grid
    grid_counts, grid_cells, cell_size,
    dt, friction, gravity
):
    forces = np.zeros_like(pos)
    torques = np.zeros_like(angle)
    
    n_rules = rule_matrices.shape[0]
    grid_h, grid_w = grid_counts.shape
    
    for i in prange(n_active):
        p_pos = pos[i]
        p_type = colors[i]
        p_angle = angle[i]
        
        # Species params
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
                    
                    # --- 1. Long-Range Matrix Forces ---
                    for r in range(n_rules):
                        # Param unpacking: min_r, max_r, strength, softening, type
                        # If max_r is 0, rule is disabled/empty
                        if rule_params[r, 1] == 0: continue
                        
                        max_r = rule_params[r, 1]
                        if dist >= max_r: continue
                        
                        factor = rule_matrices[r, p_type, colors[j]]
                        if factor == 0.0: continue
                        
                        min_r = rule_params[r, 0]
                        strength = rule_params[r, 2]
                        softening = rule_params[r, 3]
                        ftype = int(rule_params[r, 4])
                        
                        force_val = 0.0
                        
                        if ftype == 0: # LINEAR
                            if dist < min_r:
                                force_val = (dist / min_r) - 1.0
                                force_val *= 2.0
                            else:
                                range_len = max_r - min_r
                                numer = np.abs(2.0 * dist - range_len - 2.0 * min_r)
                                peak = 1.0 - (numer / range_len)
                                force_val = factor * peak
                                
                        elif ftype == 1: # INV SQUARE
                            denom = dist + softening
                            force_val = factor / (denom * denom)
                            force_val *= (1.0 - (dist/max_r)) # Fade
                            
                        elif ftype == 2: # INV CUBE
                            denom = dist + softening
                            force_val = factor / (denom * denom * denom)
                            force_val *= (1.0 - (dist/max_r))
                            
                        elif ftype == 3: # REPEL ONLY
                            force_val = factor * (1.0 - (dist / max_r))
                            
                        fx += nx_vec * force_val * strength
                        fy += ny_vec * force_val * strength

                    # --- 2. Wave Mechanics (Short Range) ---
                    # Only check if roughly within physical contact range
                    # Check max possible extent (rad + amp) * 2
                    # Heuristic: 0.2
                    if dist < 0.2:
                        o_type = colors[j]
                        o_rad = species_params[o_type, 0]
                        o_freq = species_params[o_type, 1]
                        o_amp = species_params[o_type, 2]
                        o_angle = angle[j]
                        
                        # Calculate Wave Heights at contact point
                        # Contact vector angle from i to j
                        phi_ij = np.arctan2(dy_vec, dx_vec)
                        
                        # Local angles
                        # i's surface at phi_ij
                        # j's surface at phi_ij + pi
                        
                        theta_i = phi_ij - p_angle
                        theta_j = (phi_ij + np.pi) - o_angle
                        
                        h_i = p_amp * np.cos(p_freq * theta_i)
                        h_j = o_amp * np.cos(o_freq * theta_j)
                        
                        # Actual radii at contact
                        r_i = p_rad + h_i
                        r_j = o_rad + h_j
                        
                        # Penetration
                        gap = dist - (r_i + r_j)
                        
                        if gap < 0:
                            # Repulsion
                            # Modulate strength by phase interaction
                            # Peak (h>0) vs Trough (h<0)
                            # h_i * h_j > 0 => Constructive (Peak-Peak or Trough-Trough) -> High Force
                            # h_i * h_j < 0 => Destructive (Peak-Trough) -> Low Force
                            
                            # Normalized wave values (-1 to 1)
                            # Use: 1 + (h_i/amp * h_j/amp)
                            # If amps are 0, handle safe
                            # Construct modulation factor M
                            
                            # Default M = 1.0
                            M = 1.0
                            if p_amp > 0 and o_amp > 0:
                                norm_i = h_i / p_amp
                                norm_j = h_j / o_amp
                                correlation = norm_i * norm_j
                                # If corr = 1 (Peak/Peak), M = 2.0?
                                # If corr = -1 (Peak/Trough), M = 0.5?
                                M = 1.0 + 0.8 * correlation 
                            
                            # Exponential Force
                            # F = strength * exp(-gap * k) * M
                            # gap is negative, so -gap is positive
                            # We want F to increase as gap becomes more negative
                            # exp(-gap) grows.
                            # But standard exponential potential is usually exp(-dist/scale).
                            # Here we use linear spring or exponential spring for penetration?
                            # User said "rapidly and exponentially increase... with proximity"
                            
                            # Let's use F = S * (exp(-gap * k) - 1)
                            # Note gap is negative. -gap is positive.
                            # wait, exp(positive) -> huge.
                            # As gap -> 0 (touching), F -> 0? No, touching should have some force?
                            # Usually repulsive force starts when gap < interaction_margin.
                            # Here gap < 0 is overlap.
                            # Let's use F = S * M * (exp(-gap * wave_exp) - 1.0)
                            
                            f_wave = wave_strength * M * (np.exp(-gap * wave_exp) - 1.0)
                            
                            # Apply Force
                            # Direction is repulsive (-nx_vec)
                            # Wait, nx_vec is vector FROM i TO j.
                            # Force on i should be AWAY from j (negative).
                            fx -= nx_vec * f_wave
                            fy -= ny_vec * f_wave
                            
                            # --- Torque ---
                            # Contact point relative to i center:
                            # vec_contact = r_i * nx_vec
                            # Force acts at contact point along normal?
                            # If it's a central force, no torque.
                            # But waves imply non-circular surface. Normal is not radial!
                            # Normal of surface: dR/dTheta ...
                            # Tangent vector of wave surface?
                            # Tangent angle psi = theta + pi/2 + atan( (dR/dTheta) / R )
                            # dR/dTheta = -A * f * sin(f * theta)
                            
                            # Torque = r x F.
                            # If Force is purely radial (nx_vec), Torque is 0.
                            # But collision force acts normal to the SURFACE at contact point.
                            # So we need the surface normal vector.
                            
                            # Surface Slope relative to radial:
                            # slope = (1/R) * dR/dtheta = (-A*f*sin(f*theta)) / (R0 + A*cos...)
                            # Local Normal deviation delta = atan(slope)
                            # Radial vector angle is phi_ij.
                            # Normal angle = phi_ij + delta.
                            
                            # Calculate delta for i
                            slope_i = (-p_amp * p_freq * np.sin(p_freq * theta_i)) / r_i
                            delta_i = np.arctan(slope_i)
                            
                            # Force direction on i is opposite to normal of j?
                            # Or normal of contact plane?
                            # For simplicity, assume force acts along the mean normal of the contact.
                            # Let's add a tangential component to the force based on slope difference?
                            # Or just apply torque based on the radial force component acting on the lever arm?
                            # If force is radial (pushing apart centers), but surface is tilted, it creates torque?
                            # No, if force is radial, cross product is 0.
                            # The force direction MUST be normal to surface.
                            # So Force_vec = Rot(nx_vec, delta_i) * Magnitude
                            
                            # Rotate nx_vec by delta_i
                            # But we have two surfaces. The contact normal is complex.
                            # Let's approximate: Force acts normal to i's surface.
                            # Force_on_i = -1 * Magnitude * Normal_vec_i
                            # Normal_vec_i angle = phi_ij + delta_i
                            
                            # Let's apply this torque to i.
                            # Tau = r x F.
                            # Lever arm is r_i (radial). Force angle deviates by delta_i.
                            # Cross product = |r| |F| sin(angle_between)
                            # angle_between = delta_i.
                            # Tau = r_i * f_wave * sin(delta_i)
                            
                            tau += r_i * f_wave * np.sin(delta_i) * 0.1 # Scaling
        
        forces[i, 0] = fx
        forces[i, 1] = fy - gravity
        torques[i] = tau
        
    return forces, torques

@njit(parallel=True, fastmath=True)
def integrate(pos, vel, angle, ang_vel, forces, torques, n_active, dt, friction, bounds):
    lower = bounds[0]
    upper = bounds[1]
    damp = -0.7 
    rot_friction = 0.9 # Angular drag
    
    for i in prange(n_active):
        # Linear
        vel[i, 0] = (vel[i, 0] + forces[i, 0] * dt) * friction
        vel[i, 1] = (vel[i, 1] + forces[i, 1] * dt) * friction
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        
        # Angular
        ang_vel[i] = (ang_vel[i] + torques[i] * dt) * rot_friction
        angle[i] += ang_vel[i] * dt
        
        # Normalize angle? Not strictly needed for cos/sin
        
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