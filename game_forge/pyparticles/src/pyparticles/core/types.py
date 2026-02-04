"""
Core data structures and type definitions.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from enum import Enum

class RenderMode(Enum):
    SPRITES = "sprites"  # High quality, slower (N < 5000)
    PIXELS = "pixels"    # Fast, 1px dots (N > 5000)
    GLOW = "glow"        # Additive blending (experimental)

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
    
    # Interaction
    max_radius: float = 0.1
    min_radius: float = 0.02
    repulsion_strength: float = 2.0
    
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
    """
    Structure of Arrays (SoA) container for particle data.
    Designed for Numba access.
    """
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
