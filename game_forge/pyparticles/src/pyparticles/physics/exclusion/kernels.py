"""
Eidosian PyParticles V6.2 - Enhanced Exclusion Kernels

Numba JIT-compiled kernels for quantum-inspired exclusion mechanics
with WAVE PERIMETER integration and proper SPIN dynamics.
"""

import numpy as np
from pyparticles._numba_compat import njit, prange


@njit(fastmath=True, cache=True, inline='always')
def wave_radius_at_angle(base_rad: float, freq: float, amp: float, theta: float) -> float:
    """
    Calculate wave-deformed radius at given angle.
    
    Args:
        base_rad: Base particle radius
        freq: Wave frequency (number of lobes)
        amp: Wave amplitude
        theta: Angle in radians (local to particle)
        
    Returns:
        Effective radius at that angle
    """
    return base_rad + amp * np.cos(freq * theta)


@njit(fastmath=True, cache=True, inline='always')
def compute_exclusion_force_wave(
    dist: float,
    r_i: float,
    r_j: float,
    freq_i: float,
    freq_j: float,
    amp_i: float,
    amp_j: float,
    theta_i: float,  # Angle from i to j in i's frame
    theta_j: float,  # Angle from j to i in j's frame
    spin_i: int,
    spin_j: int,
    behavior: int,
    exclusion_strength: float,
    exclusion_radius_factor: float,
) -> tuple:
    """
    Compute exclusion force between two particles using WAVE PERIMETERS.
    
    The exclusion zone is defined by the wave-deformed shapes of both particles.
    Force is calculated based on overlap of these wave perimeters.
    
    Args:
        dist: Distance between particle centers
        r_i, r_j: Base radii of particles
        freq_i, freq_j: Wave frequencies
        amp_i, amp_j: Wave amplitudes
        theta_i, theta_j: Contact angles in each particle's local frame
        spin_i, spin_j: Spin states (-1, 0, +1)
        behavior: ParticleBehavior (0=classical, 1=fermionic, 2=bosonic)
        exclusion_strength: Base exclusion force strength
        exclusion_radius_factor: Multiple of sum(radii) for max exclusion zone
        
    Returns:
        (force_magnitude, spin_torque)
        force_magnitude: Positive = repulsion
        spin_torque: Torque contribution for spin coupling
    """
    # Calculate effective radii at contact angles (using wave deformation)
    eff_r_i = wave_radius_at_angle(r_i, freq_i, amp_i, theta_i)
    eff_r_j = wave_radius_at_angle(r_j, freq_j, amp_j, theta_j)
    
    # Ensure non-negative effective radii
    eff_r_i = max(eff_r_i, r_i * 0.1)
    eff_r_j = max(eff_r_j, r_j * 0.1)
    
    # Exclusion zone based on actual wave perimeters (not just base radii)
    surface_sum = eff_r_i + eff_r_j
    exclusion_r = surface_sum * exclusion_radius_factor
    
    if dist >= exclusion_r:
        return 0.0, 0.0
    
    # Calculate penetration depth
    gap = dist - surface_sum
    
    # Classical: simple hard-sphere repulsion
    if behavior == 0:
        if gap >= 0:
            return 0.0, 0.0
        # Simple exponential repulsion for overlap
        penetration = -gap / surface_sum
        force = exclusion_strength * 0.3 * np.exp(penetration * 2.0)
        return force, 0.0
    
    # Bosonic: can condense (mild attraction at close range)
    if behavior == 2:
        if gap < 0:
            # Overlapping bosons: mild repulsion to prevent singularity
            penetration = -gap / surface_sum
            force = exclusion_strength * 0.1 * penetration
            return force, 0.0
        elif gap < surface_sum * 0.3:
            # Close bosons: slight attraction (condensation tendency)
            attraction = -0.05 * exclusion_strength * (1.0 - gap / (surface_sum * 0.3))
            return attraction, 0.0
        return 0.0, 0.0
    
    # FERMIONIC: Pauli-like exclusion with spin dependence
    spin_factor = 1.0
    spin_torque = 0.0
    
    if spin_i == spin_j and spin_i != 0:
        # SAME SPIN: Maximum exclusion (Pauli principle)
        # Cannot occupy same quantum state
        spin_factor = 2.5
        # Strong torque to flip one spin
        if gap < 0:
            spin_torque = exclusion_strength * 0.3 * np.sign(float(spin_i))
    elif spin_i == -spin_j and spin_i != 0:
        # OPPOSITE SPIN: Can form "Cooper pair" - reduced exclusion
        # Allow closer approach
        spin_factor = 0.2
        # Stabilizing torque (keep spins paired)
        spin_torque = -exclusion_strength * 0.1 * np.sign(float(spin_i))
    elif spin_i == 0 or spin_j == 0:
        # One spinless: moderate exclusion
        spin_factor = 0.6
        # Torque to align with spinning particle
        if spin_i != 0:
            spin_torque = exclusion_strength * 0.05 * float(spin_i)
        elif spin_j != 0:
            spin_torque = -exclusion_strength * 0.05 * float(spin_j)
    
    # Calculate force based on gap
    if gap < 0:
        # Overlapping: exponential repulsion
        penetration = -gap / surface_sum
        force = exclusion_strength * spin_factor * (np.exp(penetration * 3.0) - 1.0)
    else:
        # Close but not overlapping: soft exclusion field
        proximity = 1.0 - gap / exclusion_r
        force = exclusion_strength * spin_factor * 0.3 * proximity * proximity
    
    # Wave interference modulation (crests repel more, troughs less)
    if abs(amp_i) > 0.001 and abs(amp_j) > 0.001:
        # Normalized wave heights at contact
        h_i = amp_i * np.cos(freq_i * theta_i)
        h_j = amp_j * np.cos(freq_j * theta_j)
        norm_i = h_i / amp_i if abs(amp_i) > 0.001 else 0.0
        norm_j = h_j / amp_j if abs(amp_j) > 0.001 else 0.0
        
        # Both at crests (+1): enhanced repulsion
        # Both at troughs (-1): enhanced repulsion (compressed)
        # One crest, one trough: reduced repulsion
        interference = 1.0 + 0.3 * (norm_i * norm_j)
        force *= interference
    
    return force, spin_torque


@njit(fastmath=True, cache=True)
def apply_exclusion_forces_wave(
    pos: np.ndarray,           # (N, 2)
    colors: np.ndarray,        # (N,) int32 - particle types
    angle: np.ndarray,         # (N,) float32 - particle angles
    spin: np.ndarray,          # (N,) int8 - spin states
    species_params: np.ndarray,  # (T, 6) - [radius, freq, amp, inertia, spin_fric, base_spin]
    behavior_matrix: np.ndarray,  # (T, T) int32 - behavior for type pairs
    forces: np.ndarray,        # (N, 2) output forces (added to)
    torques: np.ndarray,       # (N,) output torques (added to)
    n_active: int,
    exclusion_strength: float,
    exclusion_radius_factor: float,
    grid_counts: np.ndarray,
    grid_cells: np.ndarray,
    cell_size: float,
    half_world: float,
):
    """
    OPTIMIZED: Apply exclusion forces using WAVE PERIMETERS and SPIN COUPLING.
    
    Performance optimizations:
    - Fast-path for circular particles (wave_amp ≈ 0) - no trig
    - Early distance culling before expensive calculations
    - Inlined force calculation
    """
    grid_h, grid_w = grid_counts.shape
    
    # Small threshold for "effectively zero" wave amplitude
    WAVE_THRESHOLD = 0.001
    
    for i in range(n_active):
        p_pos = pos[i]
        p_type = colors[i]
        p_angle = angle[i]
        p_spin = spin[i]
        
        p_rad = species_params[p_type, 0]
        p_freq = species_params[p_type, 1]
        p_amp = species_params[p_type, 2]
        
        # Pre-check: is this a wave particle?
        p_has_wave = abs(p_amp) > WAVE_THRESHOLD
        
        fx = 0.0
        fy = 0.0
        tau = 0.0
        
        # Grid cell
        cx = int((p_pos[0] + half_world) / cell_size)
        cy = int((p_pos[1] + half_world) / cell_size)
        
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
                    if i >= j:  # Avoid double-counting and self
                        continue
                    
                    o_pos = pos[j]
                    dx_vec = o_pos[0] - p_pos[0]
                    dy_vec = o_pos[1] - p_pos[1]
                    dist_sq = dx_vec * dx_vec + dy_vec * dy_vec
                    
                    if dist_sq < 1e-10:
                        continue
                    
                    o_type = colors[j]
                    o_rad = species_params[o_type, 0]
                    o_amp = species_params[o_type, 2]
                    
                    # EARLY CULLING: max possible exclusion radius
                    # Even with wave deformation, can't exceed base + amp
                    max_r_sum = (p_rad + abs(p_amp)) + (o_rad + abs(o_amp))
                    max_exclusion_r_sq = (max_r_sum * exclusion_radius_factor) ** 2
                    
                    if dist_sq > max_exclusion_r_sq:
                        continue
                    
                    dist = np.sqrt(dist_sq)
                    
                    o_freq = species_params[o_type, 1]
                    o_angle = angle[j]
                    o_spin = spin[j]
                    o_has_wave = abs(o_amp) > WAVE_THRESHOLD
                    
                    behavior = behavior_matrix[p_type, o_type]
                    
                    # ═══════════════════════════════════════════════════════
                    # FAST PATH: Both circular (no waves) - skip all trig
                    # ═══════════════════════════════════════════════════════
                    if not p_has_wave and not o_has_wave:
                        # Simple circular exclusion
                        eff_r_i = p_rad
                        eff_r_j = o_rad
                        surface_sum = eff_r_i + eff_r_j
                        exclusion_r = surface_sum * exclusion_radius_factor
                        
                        if dist >= exclusion_r:
                            continue
                        
                        gap = dist - surface_sum
                        
                        # Compute force based on behavior
                        # Use LINEAR scaling instead of exponential for stability
                        if behavior == 0:  # Classical
                            if gap >= 0:
                                continue
                            penetration = -gap / surface_sum
                            # Linear force instead of exponential
                            force_mag = exclusion_strength * penetration
                            spin_torque = 0.0
                        elif behavior == 2:  # Bosonic
                            if gap < 0:
                                penetration = -gap / surface_sum
                                force_mag = exclusion_strength * 0.3 * penetration
                            elif gap < surface_sum * 0.3:
                                force_mag = -0.05 * exclusion_strength * (1.0 - gap / (surface_sum * 0.3))
                            else:
                                continue
                            spin_torque = 0.0
                        else:  # Fermionic
                            # Spin factor
                            spin_factor = 1.0
                            spin_torque = 0.0
                            if p_spin == o_spin and p_spin != 0:
                                spin_factor = 2.0  # Reduced from 2.5
                                if gap < 0:
                                    spin_torque = exclusion_strength * 0.1 * float(p_spin)
                            elif p_spin == -o_spin and p_spin != 0:
                                spin_factor = 0.3  # Increased from 0.2
                                spin_torque = -exclusion_strength * 0.05 * float(p_spin)
                            elif p_spin == 0 or o_spin == 0:
                                spin_factor = 0.6
                            
                            if gap < 0:
                                penetration = -gap / surface_sum
                                # Linear force scaled by spin
                                force_mag = exclusion_strength * spin_factor * penetration
                            else:
                                proximity = 1.0 - gap / exclusion_r
                                force_mag = exclusion_strength * spin_factor * 0.2 * proximity * proximity
                    else:
                        # ═══════════════════════════════════════════════════════
                        # SLOW PATH: Wave deformation - need trig
                        # ═══════════════════════════════════════════════════════
                        phi_ij = np.arctan2(dy_vec, dx_vec)
                        theta_i = phi_ij - p_angle
                        theta_j = (phi_ij + np.pi) - o_angle
                        
                        # Wave-deformed radii
                        eff_r_i = p_rad + p_amp * np.cos(p_freq * theta_i) if p_has_wave else p_rad
                        eff_r_j = o_rad + o_amp * np.cos(o_freq * theta_j) if o_has_wave else o_rad
                        
                        eff_r_i = max(eff_r_i, p_rad * 0.1)
                        eff_r_j = max(eff_r_j, o_rad * 0.1)
                        
                        surface_sum = eff_r_i + eff_r_j
                        exclusion_r = surface_sum * exclusion_radius_factor
                        
                        if dist >= exclusion_r:
                            continue
                        
                        gap = dist - surface_sum
                        
                        # Compute force (same logic as fast path) - LINEAR scaling
                        if behavior == 0:  # Classical
                            if gap >= 0:
                                continue
                            penetration = -gap / surface_sum
                            force_mag = exclusion_strength * penetration
                            spin_torque = 0.0
                        elif behavior == 2:  # Bosonic
                            if gap < 0:
                                penetration = -gap / surface_sum
                                force_mag = exclusion_strength * 0.3 * penetration
                            elif gap < surface_sum * 0.3:
                                force_mag = -0.05 * exclusion_strength * (1.0 - gap / (surface_sum * 0.3))
                            else:
                                continue
                            spin_torque = 0.0
                        else:  # Fermionic
                            spin_factor = 1.0
                            spin_torque = 0.0
                            if p_spin == o_spin and p_spin != 0:
                                spin_factor = 2.0
                                if gap < 0:
                                    spin_torque = exclusion_strength * 0.1 * float(p_spin)
                            elif p_spin == -o_spin and p_spin != 0:
                                spin_factor = 0.3
                                spin_torque = -exclusion_strength * 0.05 * float(p_spin)
                            elif p_spin == 0 or o_spin == 0:
                                spin_factor = 0.6
                            
                            if gap < 0:
                                penetration = -gap / surface_sum
                                force_mag = exclusion_strength * spin_factor * penetration
                            else:
                                proximity = 1.0 - gap / exclusion_r
                                force_mag = exclusion_strength * spin_factor * 0.2 * proximity * proximity
                        
                        # Wave interference modulation
                        if p_has_wave and o_has_wave:
                            h_i = p_amp * np.cos(p_freq * theta_i)
                            h_j = o_amp * np.cos(o_freq * theta_j)
                            norm_i = h_i / p_amp
                            norm_j = h_j / o_amp
                            interference = 1.0 + 0.3 * (norm_i * norm_j)
                            force_mag *= interference
                    
                    if abs(force_mag) > 1e-8:
                        # Unit vector from i to j
                        inv_dist = 1.0 / dist
                        nx_vec = dx_vec * inv_dist
                        ny_vec = dy_vec * inv_dist
                        
                        # Apply forces to both particles (Newton's 3rd law)
                        fx -= nx_vec * force_mag
                        fy -= ny_vec * force_mag
                        forces[j, 0] += nx_vec * force_mag
                        forces[j, 1] += ny_vec * force_mag
                        
                        tau += spin_torque
                        torques[j] -= spin_torque
        
        forces[i, 0] += fx
        forces[i, 1] += fy
        torques[i] += tau


@njit(fastmath=True, cache=True, inline='always')
def compute_spin_coupling_torque(
    spin_i: int,
    spin_j: int,
    ang_vel_i: float,
    ang_vel_j: float,
    dist: float,
    coupling_strength: float,
    interaction_range: float,
) -> float:
    """
    Compute spin-spin coupling torque for angular velocity alignment.
    
    Particles with same spin tend to rotate in same direction.
    Particles with opposite spin tend to counter-rotate.
    
    Returns torque to apply to particle i.
    """
    if dist > interaction_range:
        return 0.0
    
    falloff = 1.0 - dist / interaction_range
    
    # Spin alignment determines coupling type
    if spin_i == spin_j and spin_i != 0:
        # Same spin: ferromagnetic coupling - align rotations
        delta_omega = ang_vel_j - ang_vel_i
        return coupling_strength * delta_omega * falloff
    elif spin_i == -spin_j and spin_i != 0:
        # Opposite spin: antiferromagnetic - counter-rotate
        sum_omega = ang_vel_i + ang_vel_j
        return -coupling_strength * sum_omega * 0.5 * falloff
    
    return 0.0


@njit(fastmath=True, cache=True)
def apply_spin_coupling(
    pos: np.ndarray,
    spin: np.ndarray,
    ang_vel: np.ndarray,
    colors: np.ndarray,
    spin_coupling_matrix: np.ndarray,  # (T, T) coupling strengths
    torques: np.ndarray,  # Output (added to)
    n_active: int,
    coupling_strength: float,
    interaction_range: float,
    grid_counts: np.ndarray,
    grid_cells: np.ndarray,
    cell_size: float,
    half_world: float,
):
    """
    OPTIMIZED: Apply spin-spin coupling torques between nearby particles.
    
    This creates emergent rotational ordering:
    - Ferromagnetic domains (same-spin regions rotate together)
    - Antiferromagnetic alternation (checkerboard rotation patterns)
    """
    grid_h, grid_w = grid_counts.shape
    interaction_range_sq = interaction_range * interaction_range
    
    for i in range(n_active):
        if spin[i] == 0:
            continue
        
        p_pos = pos[i]
        p_spin = spin[i]
        p_omega = ang_vel[i]
        p_type = colors[i]
        
        tau = 0.0
        
        cx = int((p_pos[0] + half_world) / cell_size)
        cy = int((p_pos[1] + half_world) / cell_size)
        
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
                    if i >= j or spin[j] == 0:
                        continue
                    
                    o_pos = pos[j]
                    dx_vec = o_pos[0] - p_pos[0]
                    dy_vec = o_pos[1] - p_pos[1]
                    dist_sq = dx_vec * dx_vec + dy_vec * dy_vec
                    
                    # Early distance cull
                    if dist_sq > interaction_range_sq:
                        continue
                    
                    dist = np.sqrt(dist_sq)
                    falloff = 1.0 - dist / interaction_range
                    
                    o_type = colors[j]
                    coupling = spin_coupling_matrix[p_type, o_type] * coupling_strength
                    
                    # Inline the torque calculation
                    o_spin = spin[j]
                    if p_spin == o_spin:
                        # Same spin: ferromagnetic - align rotations
                        delta_omega = ang_vel[j] - p_omega
                        torque = coupling * delta_omega * falloff
                    elif p_spin == -o_spin:
                        # Opposite spin: antiferromagnetic - counter-rotate
                        sum_omega = p_omega + ang_vel[j]
                        torque = -coupling * sum_omega * 0.5 * falloff
                    else:
                        continue
                    
                    tau += torque
                    torques[j] -= torque
        
        torques[i] += tau


@njit(parallel=True, fastmath=True, cache=True)
def apply_spin_flip(
    vel: np.ndarray,           # (N, 2) velocities
    spin: np.ndarray,          # (N,) int8 spin states (modified in-place)
    colors: np.ndarray,        # (N,) particle types
    flip_threshold: np.ndarray,    # (T,) energy threshold
    flip_probability: np.ndarray,  # (T,) flip probability
    spin_enabled: np.ndarray,      # (T,) bool
    n_active: int,
    rng_seed: int,
) -> int:
    """
    Apply stochastic spin flips based on kinetic energy.
    
    High-energy collisions can flip spins, like heating a magnet.
    
    Returns number of flips that occurred.
    """
    np.random.seed(rng_seed)
    n_flips = 0
    
    for i in prange(n_active):
        t = colors[i]
        
        if not spin_enabled[t]:
            continue
        
        # Current kinetic energy
        vx = vel[i, 0]
        vy = vel[i, 1]
        ke = 0.5 * (vx * vx + vy * vy)
        
        threshold = flip_threshold[t]
        
        if ke > threshold:
            # Probability increases with excess energy
            excess = (ke - threshold) / threshold
            prob = flip_probability[t] * (1.0 + excess)
            prob = min(prob, 0.5)  # Cap at 50%
            
            if np.random.random() < prob:
                # Flip spin
                if spin[i] == 1:
                    spin[i] = -1
                elif spin[i] == -1:
                    spin[i] = 1
                else:
                    # Initialize spinless particle
                    spin[i] = 1 if np.random.random() > 0.5 else -1
                n_flips += 1
    
    return n_flips


@njit(fastmath=True, cache=True)
def compute_spin_statistics(
    spin: np.ndarray,
    pos: np.ndarray,
    n_active: int,
    correlation_range: float = 5.0,
) -> tuple:
    """
    Compute spin distribution statistics.
    
    Returns:
        (n_up, n_down, n_none, total_spin, correlation)
    """
    n_up = 0
    n_down = 0
    n_none = 0
    
    for i in range(n_active):
        s = spin[i]
        if s > 0:
            n_up += 1
        elif s < 0:
            n_down += 1
        else:
            n_none += 1
    
    total_spin = n_up - n_down
    
    # Spatial spin correlation (expensive, sample-based)
    correlation = 0.0
    n_pairs = 0
    sample_size = min(500, n_active)
    step = max(1, n_active // sample_size)
    
    for i in range(0, n_active, step):
        if spin[i] == 0:
            continue
        for j in range(i + step, n_active, step):
            if spin[j] == 0:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < correlation_range:
                alignment = float(spin[i] * spin[j])
                falloff = 1.0 - dist / correlation_range
                correlation += alignment * falloff
                n_pairs += 1
    
    if n_pairs > 0:
        correlation /= n_pairs
    
    return n_up, n_down, n_none, total_spin, correlation


@njit(fastmath=True, cache=True)
def initialize_spins(
    spin: np.ndarray,
    colors: np.ndarray,
    spin_enabled: np.ndarray,
    n_active: int,
    seed: int = 42,
):
    """
    Initialize random spin states for particles.
    
    Spin-enabled types get random UP/DOWN.
    Non-spin types get NONE (0).
    """
    np.random.seed(seed)
    
    for i in range(n_active):
        t = colors[i]
        if spin_enabled[t]:
            spin[i] = 1 if np.random.random() > 0.5 else -1
        else:
            spin[i] = 0
