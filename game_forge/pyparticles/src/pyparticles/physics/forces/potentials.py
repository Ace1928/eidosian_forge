"""
Eidosian PyParticles V6 - Force Potentials

Numba-optimized force calculation functions for various interaction types.
All functions are JIT-compiled for maximum performance.
"""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def linear_force(
    dist: float,
    min_r: float,
    max_r: float,
    factor: float,
    strength: float,
) -> float:
    """
    Classic Particle Life linear force.
    
    Repulsive inside min_r, attractive bell curve between min_r and max_r.
    
    Args:
        dist: Distance between particles
        min_r: Repulsion threshold
        max_r: Maximum interaction range
        factor: Matrix interaction value
        strength: Global strength multiplier
        
    Returns:
        Signed force magnitude (positive = attractive)
    """
    if dist >= max_r:
        return 0.0
    
    if dist < min_r:
        # Strong repulsion inside core
        force_val = (dist / min_r) - 1.0
        force_val *= 3.0  # Repulsion multiplier
    else:
        # Bell curve attraction/repulsion
        range_len = max_r - min_r
        numer = np.abs(2.0 * dist - range_len - 2.0 * min_r)
        peak = 1.0 - (numer / range_len)
        force_val = factor * peak
    
    return force_val * strength


@njit(fastmath=True, cache=True)
def inverse_force(
    dist: float,
    max_r: float,
    factor: float,
    strength: float,
    softening: float = 0.01,
) -> float:
    """
    1/r force (magnetic-like).
    
    Args:
        dist: Distance between particles
        max_r: Cutoff radius
        factor: Interaction coefficient (positive = attractive)
        strength: Global multiplier
        softening: Prevents singularity at r=0
        
    Returns:
        Signed force magnitude
    """
    if dist >= max_r:
        return 0.0
    
    denom = dist + softening
    force_val = factor / denom
    # Smooth cutoff near max_r
    force_val *= (1.0 - dist / max_r)
    
    return force_val * strength


@njit(fastmath=True, cache=True)
def inverse_square_force(
    dist: float,
    max_r: float,
    factor: float,
    strength: float,
    softening: float = 0.01,
) -> float:
    """
    1/r² force (gravity/Coulomb-like).
    
    Returns:
        Signed force magnitude
    """
    if dist >= max_r:
        return 0.0
    
    denom = dist + softening
    force_val = factor / (denom * denom)
    force_val *= (1.0 - dist / max_r)
    
    return force_val * strength


@njit(fastmath=True, cache=True)
def inverse_cube_force(
    dist: float,
    max_r: float,
    factor: float,
    strength: float,
    softening: float = 0.01,
) -> float:
    """
    1/r³ force (dipole-dipole-like).
    
    Returns:
        Signed force magnitude
    """
    if dist >= max_r:
        return 0.0
    
    denom = dist + softening
    force_val = factor / (denom * denom * denom)
    force_val *= (1.0 - dist / max_r)
    
    return force_val * strength


@njit(fastmath=True, cache=True)
def yukawa_force(
    dist: float,
    max_r: float,
    factor: float,
    strength: float,
    decay_length: float = 0.1,
    softening: float = 0.01,
) -> float:
    """
    Yukawa (screened Coulomb) potential derivative.
    
    V(r) = (k/r) * exp(-r/λ)
    F(r) = -dV/dr = k * (1/r² + 1/λr) * exp(-r/λ)
    
    Models forces with finite range due to screening (like nuclear forces).
    
    Args:
        dist: Distance between particles
        max_r: Cutoff radius
        factor: Interaction coefficient
        strength: Global multiplier
        decay_length: Screening length λ
        softening: Prevents singularity
        
    Returns:
        Signed force magnitude
    """
    if dist >= max_r:
        return 0.0
    
    r = dist + softening
    exp_term = np.exp(-dist / decay_length)
    
    # Force = k * (1/r² + 1/(λr)) * exp(-r/λ)
    inv_r = 1.0 / r
    force_val = factor * (inv_r * inv_r + inv_r / decay_length) * exp_term
    
    return force_val * strength


@njit(fastmath=True, cache=True)
def lennard_jones_force(
    dist: float,
    max_r: float,
    factor: float,
    strength: float,
    sigma: float = 0.05,
    softening: float = 0.005,
) -> float:
    """
    Lennard-Jones 6-12 potential derivative.
    
    V(r) = 4ε * [(σ/r)¹² - (σ/r)⁶]
    F(r) = 24ε/r * [2(σ/r)¹² - (σ/r)⁶]
    
    Classic molecular dynamics potential with:
    - Strong repulsion at short range (r < σ)
    - Weak attraction at medium range
    - Zero force at r = 2^(1/6) * σ ≈ 1.122σ
    
    Args:
        dist: Distance between particles
        max_r: Cutoff radius
        factor: Well depth ε (positive)
        strength: Global multiplier
        sigma: Size parameter σ
        softening: Prevents singularity
        
    Returns:
        Signed force magnitude (positive = repulsive)
    """
    if dist >= max_r:
        return 0.0
    
    r = dist + softening
    ratio = sigma / r
    r6 = ratio ** 6
    r12 = r6 * r6
    
    # Force = 24ε/r * [2(σ/r)¹² - (σ/r)⁶]
    force_val = (24.0 * factor / r) * (2.0 * r12 - r6)
    
    return force_val * strength


@njit(fastmath=True, cache=True)
def morse_force(
    dist: float,
    max_r: float,
    factor: float,
    strength: float,
    r0: float = 0.1,
    well_width: float = 5.0,
) -> float:
    """
    Morse potential derivative.
    
    V(r) = D * [1 - exp(-a(r-r₀))]² - D
    F(r) = 2aD * exp(-a(r-r₀)) * [1 - exp(-a(r-r₀))]
         = 2aD * [exp(-a(r-r₀)) - exp(-2a(r-r₀))]
    
    Models bond-like interactions with:
    - Equilibrium at r = r₀
    - Harmonic-like near equilibrium
    - Asymptotic approach to dissociation at large r
    
    Args:
        dist: Distance between particles
        max_r: Cutoff radius
        factor: Well depth D (positive = bonding)
        strength: Global multiplier
        r0: Equilibrium bond length
        well_width: Parameter 'a' controlling well width (larger = narrower)
        
    Returns:
        Signed force magnitude (positive = repulsive, negative = attractive)
    """
    if dist >= max_r:
        return 0.0
    
    dr = dist - r0
    exp1 = np.exp(-well_width * dr)
    exp2 = np.exp(-2.0 * well_width * dr)
    
    # Force = 2aD * [exp(-a(r-r₀)) - exp(-2a(r-r₀))]
    force_val = 2.0 * well_width * factor * (exp1 - exp2)
    
    return force_val * strength


@njit(fastmath=True, cache=True)
def gaussian_force(
    dist: float,
    max_r: float,
    factor: float,
    strength: float,
    sigma: float = 0.1,
) -> float:
    """
    Gaussian-shaped soft force.
    
    F(r) = k * r * exp(-r²/2σ²) / σ²
    
    Soft, localized interaction useful for:
    - Smooth clustering
    - Avoiding harsh cutoffs
    - Modeling diffuse interactions
    
    Args:
        dist: Distance between particles
        max_r: Cutoff radius
        factor: Interaction strength
        strength: Global multiplier
        sigma: Width of Gaussian
        
    Returns:
        Signed force magnitude
    """
    if dist >= max_r:
        return 0.0
    
    sigma_sq = sigma * sigma
    exp_term = np.exp(-dist * dist / (2.0 * sigma_sq))
    
    # Derivative of Gaussian: F = k * r/σ² * exp(...)
    force_val = factor * (dist / sigma_sq) * exp_term
    
    return force_val * strength


@njit(fastmath=True, cache=True)
def compute_force_unified(
    dist: float,
    min_r: float,
    max_r: float,
    factor: float,
    strength: float,
    force_type: int,
    params: np.ndarray,
) -> float:
    """
    Unified force computation dispatch.
    
    Args:
        dist: Distance between particles
        min_r: Minimum radius (for linear force type)
        max_r: Maximum radius
        factor: Matrix interaction value
        strength: Global strength
        force_type: Integer force type code
        params: Additional parameters [softening, decay_length, sigma, r0, well_width]
        
    Returns:
        Signed force magnitude
    """
    # Extract common parameters with defaults
    softening = params[0] if len(params) > 0 else 0.01
    
    if force_type == 0:  # LINEAR
        return linear_force(dist, min_r, max_r, factor, strength)
    elif force_type == 1:  # INVERSE_SQUARE
        return inverse_square_force(dist, max_r, factor, strength, softening)
    elif force_type == 2:  # INVERSE_CUBE
        return inverse_cube_force(dist, max_r, factor, strength, softening)
    elif force_type == 3:  # REPEL_ONLY (legacy)
        if dist >= max_r:
            return 0.0
        return factor * (1.0 - dist / max_r) * strength
    elif force_type == 4:  # INVERSE
        return inverse_force(dist, max_r, factor, strength, softening)
    elif force_type == 5:  # YUKAWA
        decay = params[1] if len(params) > 1 else 0.1
        return yukawa_force(dist, max_r, factor, strength, decay, softening)
    elif force_type == 6:  # LENNARD_JONES
        sigma = params[1] if len(params) > 1 else 0.05
        return lennard_jones_force(dist, max_r, factor, strength, sigma, softening)
    elif force_type == 7:  # MORSE
        r0 = params[1] if len(params) > 1 else 0.1
        well_width = params[2] if len(params) > 2 else 5.0
        return morse_force(dist, max_r, factor, strength, r0, well_width)
    elif force_type == 8:  # GAUSSIAN
        sigma = params[1] if len(params) > 1 else 0.1
        return gaussian_force(dist, max_r, factor, strength, sigma)
    elif force_type == 9:  # EXPONENTIAL
        decay = params[1] if len(params) > 1 else 0.1
        if dist >= max_r:
            return 0.0
        return factor * np.exp(-dist / decay) * strength
    
    return 0.0
