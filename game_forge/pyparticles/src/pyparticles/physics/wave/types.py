"""
Eidosian PyParticles V6 - Wave Mechanics Types

Core data structures for wave physics simulation.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import numpy as np


class WaveMode(IntEnum):
    """Wave behavior modes."""
    DISABLED = 0       # No wave mechanics
    STANDARD = 1       # Basic wave repulsion
    INTERFERENCE = 2   # Full interference patterns
    STANDING = 3       # Standing wave formation


class WaveFeature(IntEnum):
    """Wave surface features at contact point."""
    CREST = 1          # Maximum protrusion (h = +A)
    TROUGH = -1        # Maximum indentation (h = -A)
    ZERO_RISING = 2    # Zero crossing, slope positive
    ZERO_FALLING = -2  # Zero crossing, slope negative
    SLOPE_POS = 3      # On slope, moving toward crest
    SLOPE_NEG = -3     # On slope, moving toward trough


class WaveInterference(IntEnum):
    """Interference type between two particles."""
    CONSTRUCTIVE_PEAK = 1   # Both at crest
    CONSTRUCTIVE_TROUGH = 2 # Both at trough
    DESTRUCTIVE = 3         # One crest, one trough
    QUADRATURE = 4          # 90° phase difference
    NEUTRAL = 0             # No significant interference


@dataclass
class WaveConfig:
    """
    Global wave mechanics configuration.
    
    Controls how waves affect particle interactions.
    """
    mode: WaveMode = WaveMode.STANDARD
    
    # Force modulation
    repulsion_strength: float = 30.0   # Base wave repulsion strength
    repulsion_exponent: float = 8.0    # Exponential decay rate
    
    # Interference
    constructive_multiplier: float = 1.5  # Force boost for constructive
    destructive_multiplier: float = 0.5   # Force reduction for destructive
    quadrature_torque: float = 0.3        # Torque from 90° offset
    
    # Standing waves
    standing_wave_threshold: float = 0.8  # Phase lock threshold
    standing_wave_damping: float = 0.95   # Damping toward standing
    
    # Feature detection
    crest_threshold: float = 0.9   # |h/A| > this = crest
    trough_threshold: float = 0.9  # |h/A| > this = trough
    zero_threshold: float = 0.1    # |h/A| < this = zero crossing
    
    def validate(self) -> list[str]:
        """Validate configuration and return warnings."""
        warnings = []
        if self.repulsion_strength < 0:
            warnings.append("repulsion_strength < 0 inverts wave forces")
        if self.constructive_multiplier < self.destructive_multiplier:
            warnings.append("constructive < destructive inverts interference effect")
        return warnings


@dataclass
class WaveState:
    """
    Per-particle wave state.
    
    Tracks wave phase, energy, and interaction history.
    """
    # Current phase (radians, 0 = aligned with particle angle)
    phase: np.ndarray           # (N,) float32
    
    # Phase velocity (rad/s) - can differ from species default due to interactions
    phase_velocity: np.ndarray  # (N,) float32
    
    # Standing wave partner (-1 = none)
    standing_partner: np.ndarray  # (N,) int32
    
    # Wave energy (for visualization)
    wave_energy: np.ndarray     # (N,) float32
    
    @classmethod
    def allocate(cls, max_particles: int):
        """Allocate wave state arrays."""
        return cls(
            phase=np.zeros(max_particles, dtype=np.float32),
            phase_velocity=np.zeros(max_particles, dtype=np.float32),
            standing_partner=np.full(max_particles, -1, dtype=np.int32),
            wave_energy=np.zeros(max_particles, dtype=np.float32),
        )
    
    def reset(self, n_active: int, species_phase_speeds: np.ndarray, particle_types: np.ndarray):
        """Reset wave state for particles."""
        self.phase[:n_active] = np.random.uniform(0, 2 * np.pi, n_active).astype(np.float32)
        self.phase_velocity[:n_active] = species_phase_speeds[particle_types[:n_active]]
        self.standing_partner[:n_active] = -1
        self.wave_energy[:n_active] = 0.0


@dataclass
class WaveInteraction:
    """
    Result of wave interaction calculation between two particles.
    """
    # Particle indices
    i: int
    j: int
    
    # Wave heights at contact point
    height_i: float
    height_j: float
    
    # Wave slopes at contact point
    slope_i: float
    slope_j: float
    
    # Feature types
    feature_i: WaveFeature
    feature_j: WaveFeature
    
    # Interference classification
    interference: WaveInterference
    
    # Computed force modulation
    force_multiplier: float
    
    # Torque contributions
    torque_i: float
    torque_j: float
    
    # Gap between effective surfaces
    effective_gap: float
