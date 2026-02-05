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
    WAVE = "wave" # New mode to visualize shapes

class ForceType(IntEnum):
    LINEAR = 0          # Peak at center, linear falloff
    INVERSE_SQUARE = 1  # 1/r^2
    INVERSE_CUBE = 2    # 1/r^3 (Strong force)
    REPEL_ONLY = 3      # Linear repulsion only

@dataclass
class InteractionRule:
    """Definition of a specific interaction force layer."""
    name: str
    force_type: ForceType
    matrix: np.ndarray     # (T, T)
    max_radius: float
    min_radius: float
    strength: float = 1.0
    softening: float = 0.05 
    enabled: bool = True   # Toggle support

@dataclass
class SpeciesConfig:
    """Properties for particle species."""
    # Base physical radius
    radius: np.ndarray        # (T,) float32
    # Wave properties
    wave_freq: np.ndarray     # (T,) float32 (Integer usually, e.g., 3 lobes)
    wave_amp: np.ndarray      # (T,) float32
    wave_phase_speed: np.ndarray # (T,) float32 (Auto-rotation speed)

    @classmethod
    def default(cls, n_types: int):
        return cls(
            radius=np.full(n_types, 0.05, dtype=np.float32),
            wave_freq=np.random.randint(2, 6, n_types).astype(np.float32),
            wave_amp=np.full(n_types, 0.02, dtype=np.float32),
            wave_phase_speed=np.random.uniform(-2.0, 2.0, n_types).astype(np.float32)
        )

@dataclass
class SimulationConfig:
    """Master configuration for the simulation."""
    width: int = 1200
    height: int = 1000
    
    max_particles: int = 50000
    num_particles: int = 5000
    num_types: int = 6
    dt: float = 0.01 # Lower DT for exponential forces stability
    friction: float = 0.5
    gravity: float = 0.0
    
    # Global Physics
    default_max_radius: float = 0.1
    default_min_radius: float = 0.02
    
    # Wave Mechanics
    wave_repulsion_strength: float = 50.0 # High strength for hard collisions
    wave_repulsion_exp: float = 10.0 # Exponential falloff rate
    
    threads: int = 4
    jit_cache: bool = True
    
    render_mode: RenderMode = RenderMode.WAVE
    show_fps: bool = True
    
    @classmethod
    def default(cls):
        return cls()

@dataclass
class ParticleState:
    """SoA container."""
    pos: np.ndarray        # (N, 2)
    vel: np.ndarray        # (N, 2)
    colors: np.ndarray     # (N,) int32
    
    # New Wave State
    angle: np.ndarray      # (N,) float32 (Orientation)
    ang_vel: np.ndarray    # (N,) float32 (Angular Velocity)
    
    active: int
    
    @classmethod
    def allocate(cls, max_particles: int):
        return cls(
            pos=np.zeros((max_particles, 2), dtype=np.float32),
            vel=np.zeros((max_particles, 2), dtype=np.float32),
            colors=np.zeros(max_particles, dtype=np.int32),
            angle=np.zeros(max_particles, dtype=np.float32),
            ang_vel=np.zeros(max_particles, dtype=np.float32),
            active=0
        )