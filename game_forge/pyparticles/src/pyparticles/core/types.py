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
    """
    Per-species physical properties.
    
    All arrays are shape (T,) where T = num_types.
    """
    radius: np.ndarray           # (T,) float32 - base particle radius
    
    # Wave parameters
    wave_freq: np.ndarray        # (T,) float32 - wave lobes count
    wave_amp: np.ndarray         # (T,) float32 - wave amplitude
    wave_phase_speed: np.ndarray # (T,) float32 - wave rotation speed
    
    # Spin dynamics (NEW)
    spin_inertia: np.ndarray     # (T,) float32 - moment of inertia multiplier
    spin_friction: np.ndarray    # (T,) float32 - angular friction coefficient
    base_spin_rate: np.ndarray   # (T,) float32 - intrinsic spin tendency
    
    @classmethod
    def default(cls, n_types: int, world_size: float = 2.0):
        """Create default species config scaled to world size."""
        # Scale radius to world size (visible but not dominant)
        base_radius = 0.01 * world_size  # 1% of world size
        
        return cls(
            radius=np.full(n_types, base_radius, dtype=np.float32) * 
                   np.random.uniform(0.5, 1.5, n_types).astype(np.float32),
            wave_freq=np.random.randint(2, 8, n_types).astype(np.float32),
            wave_amp=np.full(n_types, base_radius * 0.3, dtype=np.float32) *
                     np.random.uniform(0.2, 1.5, n_types).astype(np.float32),
            wave_phase_speed=np.random.uniform(-3.0, 3.0, n_types).astype(np.float32),
            # Spin dynamics
            spin_inertia=np.random.uniform(0.5, 2.0, n_types).astype(np.float32),
            spin_friction=np.random.uniform(0.5, 3.0, n_types).astype(np.float32),
            base_spin_rate=np.random.uniform(-2.0, 2.0, n_types).astype(np.float32),
        )


@dataclass 
class SpinInteractionMatrix:
    """
    Species-to-species spin interaction matrix.
    
    Determines how particle types influence each other's rotation:
    - Positive values: aligned spin coupling (same direction)
    - Negative values: counter-rotation coupling
    - Zero: no spin interaction
    """
    matrix: np.ndarray  # (T, T) float32 - spin coupling strength
    
    @classmethod
    def default(cls, n_types: int):
        """Create randomized spin interaction matrix."""
        mat = np.random.uniform(-1.0, 1.0, (n_types, n_types)).astype(np.float32)
        # Make symmetric for Newton's 3rd law
        mat = (mat + mat.T) / 2
        return cls(matrix=mat)
    
    @classmethod
    def diagonal(cls, n_types: int, coupling: float = 0.5):
        """Same-type spin coupling only."""
        mat = np.eye(n_types, dtype=np.float32) * coupling
        return cls(matrix=mat)
    
    def resize(self, n_types: int):
        """Resize matrix for new type count."""
        old_n = self.matrix.shape[0]
        if n_types == old_n:
            return
        new_mat = np.zeros((n_types, n_types), dtype=np.float32)
        min_n = min(old_n, n_types)
        new_mat[:min_n, :min_n] = self.matrix[:min_n, :min_n]
        if n_types > old_n:
            # Randomize new species
            new_mat[old_n:, :] = np.random.uniform(-1, 1, (n_types - old_n, n_types))
            new_mat[:, old_n:] = np.random.uniform(-1, 1, (n_types, n_types - old_n))
        self.matrix = new_mat.astype(np.float32)

@dataclass
class SimulationConfig:
    """
    Master simulation configuration.
    
    Physical parameters scale with world_size:
    - Domain is [-world_size/2, world_size/2] x [-world_size/2, world_size/2]
    - Interaction radii, particle sizes scale proportionally
    """
    # Display
    width: int = 1400
    height: int = 1000
    
    # World geometry - EXPANDED for emergent dynamics
    world_size: float = 10.0  # Total domain size (was 2.0)
    
    # Particles - increased for emergence
    max_particles: int = 100000
    num_particles: int = 8000
    num_types: int = 12  # More species = richer dynamics
    
    # Time integration
    dt: float = 0.004  # Scaled for larger world
    substeps: int = 2  # Multiple substeps for stability
    
    # Damping - scaled to world
    friction: float = 0.3  # Linear velocity damping
    angular_friction: float = 1.5  # Angular damping
    
    gravity: float = 0.0
    
    # Interaction radii - scaled to world_size
    # These are fractions of world_size for portability
    default_max_radius_frac: float = 0.03  # 3% of world = 0.3 in 10-unit world
    default_min_radius_frac: float = 0.004  # 0.4% of world = 0.04 in 10-unit world
    
    # Wave mechanics
    wave_repulsion_strength: float = 30.0 
    wave_repulsion_exp: float = 8.0
    
    # Spin dynamics (NEW)
    spin_coupling_strength: float = 1.0  # Global spin interaction multiplier
    spin_torque_scale: float = 0.5  # How much forces create torque
    
    # Thermostat (Berendsen)
    target_temperature: float = 0.3  # Lower for more structure
    thermostat_coupling: float = 0.05  # Gentler coupling
    thermostat_enabled: bool = True
    
    # Performance
    threads: int = 8
    jit_cache: bool = True
    
    # Rendering
    render_mode: RenderMode = RenderMode.OPENGL
    show_fps: bool = True
    particle_scale: float = 1.0  # Visual scale multiplier
    
    # Computed properties
    @property
    def default_max_radius(self) -> float:
        """Absolute max interaction radius."""
        return self.default_max_radius_frac * self.world_size
    
    @property
    def default_min_radius(self) -> float:
        """Absolute min interaction radius."""
        return self.default_min_radius_frac * self.world_size
    
    @property
    def half_world(self) -> float:
        """Half the world size for bounds."""
        return self.world_size / 2.0
    
    @classmethod
    def default(cls):
        return cls()
    
    @classmethod
    def small_world(cls):
        """Small world for testing/debugging."""
        return cls(
            world_size=4.0,
            num_particles=500,
            num_types=4,
        )
    
    @classmethod
    def large_world(cls):
        """Large world for maximum emergence."""
        return cls(
            world_size=20.0,
            num_particles=20000,
            num_types=16,
            dt=0.003,
            substeps=3,
        )
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []
        if self.dt > 0.01:
            warnings.append(f"dt={self.dt} is large; consider dt<=0.005 for stability")
        if self.num_particles > self.max_particles:
            warnings.append(f"num_particles > max_particles; clamping")
            self.num_particles = self.max_particles
        if self.thermostat_coupling > 0.5:
            warnings.append(f"thermostat_coupling > 0.5 may overdamp emergent structures")
        if self.num_types < 3:
            warnings.append(f"num_types < 3 limits emergent complexity")
        if self.world_size < 2.0:
            warnings.append(f"world_size < 2.0 may cause crowding")
        return warnings

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
