"""
Eidosian PyParticles V6 - Wave Mechanics Kernels

Numba-compiled kernels for wave physics calculations.
"""

import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True, inline='always')
def compute_wave_height(theta: float, freq: float, amp: float) -> float:
    """
    Compute wave height at angle theta.
    
    h(θ) = A * cos(f * θ)
    
    Args:
        theta: Local angle from particle center (radians)
        freq: Wave frequency (number of lobes)
        amp: Wave amplitude
        
    Returns:
        Height displacement at theta
    """
    return amp * np.cos(freq * theta)


@njit(fastmath=True, cache=True, inline='always')
def compute_wave_derivative(theta: float, freq: float, amp: float) -> float:
    """
    Compute wave slope (derivative) at angle theta.
    
    dh/dθ = -A * f * sin(f * θ)
    
    Args:
        theta: Local angle from particle center (radians)
        freq: Wave frequency (number of lobes)
        amp: Wave amplitude
        
    Returns:
        Slope at theta (positive = increasing toward crest)
    """
    return -amp * freq * np.sin(freq * theta)


@njit(fastmath=True, cache=True, inline='always')
def detect_wave_feature(height: float, slope: float, amp: float, 
                        crest_thresh: float = 0.9,
                        zero_thresh: float = 0.1) -> int:
    """
    Classify wave feature at a point.
    
    Args:
        height: Wave height at point
        slope: Wave slope at point
        amp: Wave amplitude (for normalization)
        crest_thresh: Threshold for crest/trough detection
        zero_thresh: Threshold for zero crossing detection
        
    Returns:
        Feature code: 1=crest, -1=trough, 2=zero_rising, -2=zero_falling, 3=slope_pos, -3=slope_neg
    """
    if amp < 1e-6:
        return 0  # No wave
    
    normalized = height / amp
    
    if normalized > crest_thresh:
        return 1   # CREST
    elif normalized < -crest_thresh:
        return -1  # TROUGH
    elif abs(normalized) < zero_thresh:
        if slope > 0:
            return 2   # ZERO_RISING
        else:
            return -2  # ZERO_FALLING
    else:
        if slope > 0:
            return 3   # SLOPE_POS
        else:
            return -3  # SLOPE_NEG


@njit(fastmath=True, cache=True, inline='always')
def compute_interference(feature_i: int, feature_j: int,
                        height_i: float, height_j: float,
                        amp_i: float, amp_j: float) -> tuple:
    """
    Compute interference type and force multiplier.
    
    Args:
        feature_i, feature_j: Wave features at contact points
        height_i, height_j: Wave heights at contact points
        amp_i, amp_j: Wave amplitudes
        
    Returns:
        (interference_type, force_multiplier)
        interference_type: 0=neutral, 1=constructive_peak, 2=constructive_trough, 3=destructive, 4=quadrature
    """
    if amp_i < 1e-6 or amp_j < 1e-6:
        return 0, 1.0  # NEUTRAL
    
    # Normalize heights
    norm_i = height_i / amp_i
    norm_j = height_j / amp_j
    
    # Product determines interference type
    product = norm_i * norm_j
    
    if product > 0.5:
        # Same sign, both high magnitude = constructive
        if norm_i > 0 and norm_j > 0:
            return 1, 1.5  # CONSTRUCTIVE_PEAK
        else:
            return 2, 1.5  # CONSTRUCTIVE_TROUGH
    elif product < -0.5:
        # Opposite signs = destructive
        return 3, 0.5  # DESTRUCTIVE
    elif abs(product) < 0.2:
        # Near zero product, one is near zero
        return 4, 1.0  # QUADRATURE (generates torque)
    else:
        return 0, 1.0  # NEUTRAL


@njit(fastmath=True, cache=True, inline='always')
def compute_wave_force(gap: float, multiplier: float, 
                       strength: float, exponent: float) -> float:
    """
    Compute wave repulsion force magnitude.
    
    F = strength * multiplier * (exp(-gap * exponent) - 1)
    
    Only applies when gap < 0 (overlapping effective surfaces).
    
    Args:
        gap: Distance between effective surfaces (negative = overlap)
        multiplier: Interference-based force modifier
        strength: Base wave repulsion strength
        exponent: Exponential decay rate
        
    Returns:
        Force magnitude (positive = repulsion)
    """
    if gap >= 0:
        return 0.0
    
    # Clamp gap to prevent numerical explosion
    clamped_gap = max(gap, -0.3)
    
    return strength * multiplier * (np.exp(-clamped_gap * exponent) - 1.0)


@njit(fastmath=True, cache=True, inline='always')
def compute_wave_torque(force: float, slope: float, radius: float,
                        torque_scale: float = 0.5) -> float:
    """
    Compute torque from wave force on sloped surface.
    
    τ = F * slope * r * scale
    
    The slope creates a tangential component of the normal force.
    
    Args:
        force: Normal force magnitude
        slope: Wave slope at contact point
        radius: Effective radius at contact
        torque_scale: Global torque scaling
        
    Returns:
        Torque (positive = counterclockwise)
    """
    if abs(radius) < 1e-6:
        return 0.0
    
    # Normalize slope by radius for angle
    normalized_slope = slope / radius
    
    return force * normalized_slope * radius * torque_scale


@njit(parallel=True, fastmath=True, cache=True)
def update_wave_phases(phase: np.ndarray, phase_velocity: np.ndarray,
                       n_active: int, dt: float):
    """
    Update wave phases for all particles.
    
    Args:
        phase: (N,) wave phases
        phase_velocity: (N,) phase velocities
        n_active: Number of active particles
        dt: Time step
    """
    for i in prange(n_active):
        phase[i] += phase_velocity[i] * dt
        # Wrap to [0, 2π)
        while phase[i] >= 2.0 * np.pi:
            phase[i] -= 2.0 * np.pi
        while phase[i] < 0:
            phase[i] += 2.0 * np.pi


@njit(parallel=True, fastmath=True, cache=True)
def compute_wave_energy(phase: np.ndarray, phase_velocity: np.ndarray,
                        species_amp: np.ndarray, particle_types: np.ndarray,
                        n_active: int) -> np.ndarray:
    """
    Compute wave kinetic energy for visualization.
    
    E_wave = 0.5 * A² * ω²
    
    Args:
        phase: (N,) wave phases
        phase_velocity: (N,) phase velocities  
        species_amp: (T,) wave amplitudes per species
        particle_types: (N,) particle type indices
        n_active: Number of active particles
        
    Returns:
        (N,) wave energies
    """
    energy = np.zeros(n_active, dtype=np.float32)
    
    for i in prange(n_active):
        amp = species_amp[particle_types[i]]
        omega = phase_velocity[i]
        energy[i] = 0.5 * amp * amp * omega * omega
    
    return energy


@njit(fastmath=True, cache=True)
def detect_standing_wave_pairs(pos: np.ndarray, angle: np.ndarray,
                               phase: np.ndarray, colors: np.ndarray,
                               species_freq: np.ndarray, species_amp: np.ndarray,
                               n_active: int, threshold: float = 0.8,
                               max_dist: float = 0.2) -> np.ndarray:
    """
    Detect particle pairs forming standing waves.
    
    Standing waves form when:
    1. Particles are within interaction range
    2. Phases are locked (relative phase stable)
    3. Frequencies match (same or harmonic)
    
    Args:
        pos: (N, 2) positions
        angle: (N,) particle angles
        phase: (N,) wave phases
        colors: (N,) particle types
        species_freq: (T,) wave frequencies
        species_amp: (T,) wave amplitudes
        n_active: Number of active particles
        threshold: Phase lock threshold
        max_dist: Maximum distance for standing wave
        
    Returns:
        (N,) partner indices (-1 = no standing wave)
    """
    partners = np.full(n_active, -1, dtype=np.int32)
    
    for i in range(n_active):
        if partners[i] >= 0:
            continue  # Already paired
        
        freq_i = species_freq[colors[i]]
        amp_i = species_amp[colors[i]]
        
        if amp_i < 1e-6:
            continue
        
        best_score = -1.0
        best_j = -1
        
        for j in range(i + 1, n_active):
            if partners[j] >= 0:
                continue
            
            # Distance check
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            
            if dist > max_dist:
                continue
            
            freq_j = species_freq[colors[j]]
            amp_j = species_amp[colors[j]]
            
            if amp_j < 1e-6:
                continue
            
            # Frequency compatibility (same or 2:1 harmonic)
            freq_ratio = freq_i / freq_j if freq_j > freq_i else freq_j / freq_i
            if abs(freq_ratio - 1.0) > 0.1 and abs(freq_ratio - 0.5) > 0.1:
                continue
            
            # Phase lock score (0 or π relative phase = standing)
            relative_phase = abs(phase[i] - phase[j])
            if relative_phase > np.pi:
                relative_phase = 2 * np.pi - relative_phase
            
            # Score: high when relative phase is 0 or π
            phase_score = abs(np.cos(relative_phase))
            
            if phase_score > threshold and phase_score > best_score:
                best_score = phase_score
                best_j = j
        
        if best_j >= 0:
            partners[i] = best_j
            partners[best_j] = i
    
    return partners
