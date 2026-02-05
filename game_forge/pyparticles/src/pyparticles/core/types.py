"""
Core data structures and type definitions.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from enum import IntEnum, Enum

class RenderMode(Enum):
    SPRITES = "sprites"
    PIXELS = "pixels"
    GLOW = "glow"

class ForceType(IntEnum):
    LINEAR = 0          # Standard Particle Life: Peak at center, linear falloff
    INVERSE_SQUARE = 1  # Gravity/Electrostatics: 1/r^2 (softened)
    REPEL_ONLY = 2      # Hard collision only

@dataclass
class InteractionRule:
    """Definition of a specific interaction force layer."""
    name: str
    force_type: ForceType
    matrix: np.ndarray     # (T, T)
    max_radius: float
    min_radius: float
    strength: float = 1.0
    softening: float = 0.05 # For inverse square to prevent singularity

@dataclass
class SimulationConfig:
    """Master configuration for the simulation."""
    # Dimensions
    width: int = 1200
    height: int = 1000
    
    # Physics
    max_particles: int = 50000
    num_particles: int = 5000
    num_types: int = 6
    dt: float = 0.02
    friction: float = 0.5
    gravity: float = 0.0
    
    # Global Physics properties (legacy/global)
    # Individual rules now handle radii, but we keep defaults for init
    default_max_radius: float = 0.1
    default_min_radius: float = 0.02
    
    # System
    threads: int = 4
    jit_cache: bool = True
    
    # Visuals
    render_mode: RenderMode = RenderMode.SPRITES
    show_fps: bool = True
    
    @classmethod
    def default(cls):
        return cls()

@dataclass
class ParticleState:
    """Structure of Arrays (SoA) container for particle data."""
    pos: np.ndarray        # (N, 2) float32
    vel: np.ndarray        # (N, 2) float32
    colors: np.ndarray     # (N,) int32
    active: int            # Number of active particles
    
    @classmethod
    def allocate(cls, max_particles: int):
        return cls(
            pos=np.zeros((max_particles, 2), dtype=np.float32),
            vel=np.zeros((max_particles, 2), dtype=np.float32),
            colors=np.zeros(max_particles, dtype=np.int32),
            active=0
        )
