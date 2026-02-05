"""
Eidosian PyParticles V6 - Exclusion Kernels

Numba JIT-compiled kernels for quantum-inspired exclusion mechanics.
"""

import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True, inline='always')
def compute_exclusion_force(
    dist: float,
    r_i: float,
    r_j: float,
    spin_i: int,
    spin_j: int,
    behavior: int,
    exclusion_strength: float,
    exclusion_radius_factor: float,
) -> float:
    """
    Compute exclusion force between two particles.
    
    Args:
        dist: Distance between particles
        r_i, r_j: Radii of particles
        spin_i, spin_j: Spin states (-1, 0, +1)
        behavior: ParticleBehavior (0=classical, 1=fermionic, 2=bosonic)
        exclusion_strength: Base exclusion force strength
        exclusion_radius_factor: Multiple of sum(radii) for exclusion zone
        
    Returns:
        Force magnitude (positive = repulsion)
    """
    # Exclusion zone radius
    r_sum = r_i + r_j
    exclusion_r = r_sum * exclusion_radius_factor
    
    if dist >= exclusion_r:
        return 0.0
    
    # Classical: no special exclusion
    if behavior == 0:
        return 0.0
    
    # Bosonic: no exclusion (can even attract slightly)
    if behavior == 2:
        # Slight attraction for bosons at close range
        if dist < r_sum:
            return -0.1 * exclusion_strength * (1.0 - dist / r_sum)
        return 0.0
    
    # FERMIONIC: Pauli-like exclusion
    # Same spin = strong repulsion
    # Opposite spin = can pair (reduced repulsion)
    
    spin_factor = 1.0
    
    if spin_i == spin_j and spin_i != 0:
        # Same spin: MAXIMUM exclusion (Pauli principle)
        spin_factor = 2.0
    elif spin_i == -spin_j and spin_i != 0:
        # Opposite spin: Can form "Cooper pair" - reduced exclusion
        spin_factor = 0.3
    elif spin_i == 0 or spin_j == 0:
        # One spinless: moderate exclusion
        spin_factor = 0.7
    
    # Exponential repulsion
    penetration = (exclusion_r - dist) / exclusion_r
    force = exclusion_strength * spin_factor * np.exp(penetration * 3.0 - 1.0)
    
    return force


@njit(parallel=True, fastmath=True, cache=True)
def apply_exclusion_forces(
    pos: np.ndarray,           # (N, 2)
    colors: np.ndarray,        # (N,) int32 - particle types
    spin: np.ndarray,          # (N,) int8 - spin states
    radii: np.ndarray,         # (T,) float32 - species radii
    behavior_matrix: np.ndarray,  # (T, T) int32 - behavior for type pairs
    forces: np.ndarray,        # (N, 2) output forces (added to)
    n_active: int,
    exclusion_strength: float,
    exclusion_radius_factor: float,
    grid_counts: np.ndarray,
    grid_cells: np.ndarray,
    cell_size: float,
    half_world: float,
):
    """
    Apply exclusion forces to all particles using spatial grid.
    
    Modifies `forces` array in-place by ADDING exclusion forces.
    """
    grid_h, grid_w = grid_counts.shape
    
    for i in prange(n_active):
        p_pos = pos[i]
        p_type = colors[i]
        p_spin = spin[i]
        p_rad = radii[p_type]
        
        fx = 0.0
        fy = 0.0
        
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
                    
                    dist = np.sqrt(dist_sq)
                    
                    o_type = colors[j]
                    o_spin = spin[j]
                    o_rad = radii[o_type]
                    
                    behavior = behavior_matrix[p_type, o_type]
                    
                    force_mag = compute_exclusion_force(
                        dist, p_rad, o_rad,
                        p_spin, o_spin,
                        behavior,
                        exclusion_strength,
                        exclusion_radius_factor
                    )
                    
                    if abs(force_mag) > 1e-8:
                        # Unit vector from i to j
                        nx_vec = dx_vec / dist
                        ny_vec = dy_vec / dist
                        
                        # Apply to both particles (Newton's 3rd law)
                        fx += nx_vec * force_mag
                        fy += ny_vec * force_mag
                        forces[j, 0] -= nx_vec * force_mag
                        forces[j, 1] -= ny_vec * force_mag
        
        forces[i, 0] += fx
        forces[i, 1] += fy


@njit(fastmath=True, cache=True, inline='always')
def compute_spin_interaction(
    spin_i: int,
    spin_j: int,
    dist: float,
    coupling_i: float,
    coupling_j: float,
    interaction_range: float,
) -> float:
    """
    Compute spin-spin interaction energy/force modifier.
    
    Aligned spins: ferromagnetic coupling (reduce force)
    Anti-aligned spins: antiferromagnetic (increase force)
    
    Returns modifier in range [-1, +1].
    """
    if spin_i == 0 or spin_j == 0:
        return 0.0
    
    if dist > interaction_range:
        return 0.0
    
    # Coupling strength
    coupling = (coupling_i + coupling_j) * 0.5
    
    # Distance falloff
    falloff = 1.0 - dist / interaction_range
    
    # Spin alignment
    alignment = float(spin_i * spin_j)  # +1 aligned, -1 anti-aligned
    
    return alignment * coupling * falloff


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
