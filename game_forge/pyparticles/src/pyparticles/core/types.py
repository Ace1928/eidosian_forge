"""
Core data structures and type definitions.
Refactored for Procedural Physics (Thermostat, No Biology).
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from enum import IntEnum, Enum

class RenderMode(Enum):
    SPRITES = "sprites"
    PIXELS = "pixels"
    GLOW = "glow"
    WAVE = "wave"
    OPENGL = "opengl"

class ForceType(IntEnum):
    LINEAR = 0          
    INVERSE_SQUARE = 1  
    INVERSE_CUBE = 2    
    REPEL_ONLY = 3      

@dataclass
class InteractionRule:
    name: str
    force_type: ForceType
    matrix: np.ndarray     
    max_radius: float
    min_radius: float
    strength: float = 1.0
    softening: float = 0.05 
    enabled: bool = True

@dataclass
class SpeciesConfig:
    radius: np.ndarray        # (T,) float32
    # Wave Params
    wave_freq: np.ndarray     # (T,) float32
    wave_amp: np.ndarray      # (T,) float32
    wave_phase_speed: np.ndarray # (T,) float32

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
    width: int = 1200
    height: int = 1000
    
    max_particles: int = 50000
    num_particles: int = 5000
    num_types: int = 6
    dt: float = 0.01 
    friction: float = 0.0 # Friction removed? No, friction implies energy loss. Thermostat replaces it?
                          # Usually Thermostat + Conservative forces = NVT.
                          # But we want "medium" drag?
                          # Let's keep friction but set default to 1.0 (no drag) or small drag?
                          # If we have drag, thermostat pumps energy in.
                          # Let's keep a small "Langevin-like" drag or just user controllable.
    
    gravity: float = 0.0
    
    default_max_radius: float = 0.1
    default_min_radius: float = 0.02
    
    wave_repulsion_strength: float = 50.0 
    wave_repulsion_exp: float = 10.0 
    
    # Thermostat
    target_temperature: float = 0.5 # Target Kinetic Energy per particle
    thermostat_coupling: float = 0.1 # Coupling strength (0.0 to 1.0)
    
    threads: int = 4
    jit_cache: bool = True
    
    render_mode: RenderMode = RenderMode.OPENGL
    show_fps: bool = True
    
    @classmethod
    def default(cls):
        return cls()

@dataclass
class ParticleState:
    pos: np.ndarray        # (N, 2)
    vel: np.ndarray        # (N, 2)
    colors: np.ndarray     # (N,) int32
    angle: np.ndarray      # (N,) float32
    ang_vel: np.ndarray    # (N,) float32
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
