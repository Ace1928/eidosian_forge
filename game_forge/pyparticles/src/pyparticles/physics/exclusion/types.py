"""
Eidosian PyParticles V6 - Exclusion Types

Type definitions for quantum-inspired exclusion mechanics.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import numpy as np


class SpinState(IntEnum):
    """
    Discrete spin quantum number.
    
    Like electron spin: +1/2 (UP) or -1/2 (DOWN).
    Determines pairing behavior.
    """
    DOWN = -1
    NONE = 0   # For spinless particles (bosons)
    UP = 1


class ParticleBehavior(IntEnum):
    """
    Quantum statistics behavior mode.
    
    FERMIONIC: Same-type, same-spin particles strongly repel (exclusion).
    BOSONIC: Same-type particles can overlap/condense.
    CLASSICAL: No special quantum effects.
    """
    CLASSICAL = 0
    FERMIONIC = 1
    BOSONIC = 2


@dataclass
class SpinConfig:
    """
    Spin dynamics configuration per particle type.
    
    Each species can have different spin properties:
    - spin_enabled: Whether this type has spin
    - flip_threshold: Energy required for spin flip
    - flip_probability: Probability of flip when threshold exceeded
    - coupling_strength: How strongly spin affects interactions
    """
    spin_enabled: np.ndarray      # (T,) bool - does this type have spin
    flip_threshold: np.ndarray    # (T,) float32 - energy for flip
    flip_probability: np.ndarray  # (T,) float32 - flip chance
    coupling_strength: np.ndarray # (T,) float32 - spin coupling
    
    @classmethod
    def default(cls, n_types: int):
        """Create default spin config with all types spin-enabled."""
        return cls(
            spin_enabled=np.ones(n_types, dtype=np.bool_),
            flip_threshold=np.full(n_types, 2.0, dtype=np.float32),
            flip_probability=np.full(n_types, 0.1, dtype=np.float32),
            coupling_strength=np.ones(n_types, dtype=np.float32),
        )
    
    @classmethod
    def mixed(cls, n_types: int):
        """Create mixed config: some fermionic, some bosonic."""
        enabled = np.zeros(n_types, dtype=np.bool_)
        # Half the types have spin (fermionic behavior)
        enabled[:n_types//2] = True
        
        return cls(
            spin_enabled=enabled,
            flip_threshold=np.random.uniform(1.0, 3.0, n_types).astype(np.float32),
            flip_probability=np.random.uniform(0.05, 0.2, n_types).astype(np.float32),
            coupling_strength=np.random.uniform(0.5, 1.5, n_types).astype(np.float32),
        )


@dataclass
class ExclusionConfig:
    """
    Global exclusion mechanics configuration.
    
    Controls how Pauli-like exclusion affects the simulation:
    - exclusion_strength: Base repulsion strength for same-spin overlap
    - exclusion_radius_factor: Multiple of particle radius for exclusion zone
    - behavior_matrix: (T, T) matrix of ParticleBehavior for type pairs
    - allow_spin_flips: Whether particles can flip spin under stress
    """
    exclusion_strength: float = 20.0
    exclusion_radius_factor: float = 2.0
    allow_spin_flips: bool = True
    
    # Per-type behavior (T,) - FERMIONIC, BOSONIC, or CLASSICAL
    type_behavior: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Type-to-type interaction behavior matrix (T, T)
    behavior_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def initialize(self, n_types: int):
        """Initialize matrices for n_types species."""
        if len(self.type_behavior) != n_types:
            # Default: alternating fermionic/bosonic
            self.type_behavior = np.zeros(n_types, dtype=np.int32)
            for i in range(n_types):
                # 60% fermionic, 40% bosonic/classical
                if i % 5 < 3:
                    self.type_behavior[i] = ParticleBehavior.FERMIONIC
                elif i % 5 < 4:
                    self.type_behavior[i] = ParticleBehavior.BOSONIC
                else:
                    self.type_behavior[i] = ParticleBehavior.CLASSICAL
        
        if self.behavior_matrix.shape != (n_types, n_types):
            # Build matrix from type behaviors
            self.behavior_matrix = np.zeros((n_types, n_types), dtype=np.int32)
            for i in range(n_types):
                for j in range(n_types):
                    # Use more restrictive behavior
                    bi = self.type_behavior[i]
                    bj = self.type_behavior[j]
                    if bi == ParticleBehavior.FERMIONIC or bj == ParticleBehavior.FERMIONIC:
                        self.behavior_matrix[i, j] = ParticleBehavior.FERMIONIC
                    elif bi == ParticleBehavior.BOSONIC and bj == ParticleBehavior.BOSONIC:
                        self.behavior_matrix[i, j] = ParticleBehavior.BOSONIC
                    else:
                        self.behavior_matrix[i, j] = ParticleBehavior.CLASSICAL
    
    @classmethod
    def all_fermionic(cls, n_types: int, strength: float = 25.0):
        """Create config where all particles are fermionic."""
        cfg = cls(exclusion_strength=strength)
        cfg.type_behavior = np.full(n_types, ParticleBehavior.FERMIONIC, dtype=np.int32)
        cfg.behavior_matrix = np.full((n_types, n_types), ParticleBehavior.FERMIONIC, dtype=np.int32)
        return cfg
    
    @classmethod
    def all_bosonic(cls, n_types: int):
        """Create config where all particles are bosonic."""
        cfg = cls(exclusion_strength=0.0)
        cfg.type_behavior = np.full(n_types, ParticleBehavior.BOSONIC, dtype=np.int32)
        cfg.behavior_matrix = np.full((n_types, n_types), ParticleBehavior.BOSONIC, dtype=np.int32)
        return cfg


@dataclass
class SpinStatistics:
    """
    Statistics about spin states in the simulation.
    """
    n_up: int = 0
    n_down: int = 0
    n_none: int = 0
    n_flips_this_frame: int = 0
    total_spin: int = 0  # Net spin (up - down)
    spin_correlation: float = 0.0  # Spatial correlation of aligned spins
    
    def __repr__(self):
        return (f"SpinStats(up={self.n_up}, down={self.n_down}, "
                f"net={self.total_spin}, flips={self.n_flips_this_frame})")
