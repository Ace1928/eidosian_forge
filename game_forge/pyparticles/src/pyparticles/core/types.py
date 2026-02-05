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
    def default(cls, n_types: int, world_size: float = 100.0):
        """Create default species config scaled to world size."""
        # Scale radius to world size - MUCH SMALLER (0.1% of world)
        base_radius = 0.001 * world_size  # 0.1% of world size = tiny particles
        
        return cls(
            radius=np.full(n_types, base_radius, dtype=np.float32) * 
                   np.random.uniform(0.8, 1.2, n_types).astype(np.float32),
            wave_freq=np.random.randint(3, 6, n_types).astype(np.float32),
            wave_amp=np.full(n_types, base_radius * 0.2, dtype=np.float32) *
                     np.random.uniform(0.5, 1.0, n_types).astype(np.float32),
            wave_phase_speed=np.random.uniform(-2.0, 2.0, n_types).astype(np.float32),
            # Spin dynamics
            spin_inertia=np.random.uniform(0.8, 1.5, n_types).astype(np.float32),
            spin_friction=np.random.uniform(1.0, 3.0, n_types).astype(np.float32),
            base_spin_rate=np.random.uniform(-1.5, 1.5, n_types).astype(np.float32),
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
    
    # World geometry - MASSIVELY EXPANDED for emergent dynamics
    world_size: float = 100.0  # Total domain size (10x larger than before)
    
    # Particles - increased for emergence
    max_particles: int = 100000
    num_particles: int = 10000
    num_types: int = 16  # More species = richer dynamics
    
    # Time integration
    dt: float = 0.005  # Scaled for larger world
    substeps: int = 2  # Multiple substeps for stability
    
    # Damping - INCREASED for energy stability
    friction: float = 0.5  # Linear velocity damping (prevents runaway)
    angular_friction: float = 2.0  # Angular damping
    
    gravity: float = 0.0
    
    # Interaction radii - scaled to world_size
    # These are fractions of world_size for portability
    default_max_radius_frac: float = 0.02  # 2% of world = 2.0 in 100-unit world
    default_min_radius_frac: float = 0.002  # 0.2% of world = 0.2 in 100-unit world
    
    # Wave mechanics
    wave_repulsion_strength: float = 50.0  # STRONGER repulsion
    wave_repulsion_exp: float = 6.0  # Softer falloff for wider effect
    
    # Spin dynamics (NEW)
    spin_coupling_strength: float = 1.0  # Global spin interaction multiplier
    spin_torque_scale: float = 0.5  # How much forces create torque
    
    # Thermostat (Berendsen) - CRITICAL for energy stability
    target_temperature: float = 0.5  # Target kinetic energy
    thermostat_coupling: float = 0.1  # STRONGER coupling to cap energy
    thermostat_enabled: bool = True
    max_velocity: float = 10.0  # Hard cap on velocity magnitude
    
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
            world_size=20.0,
            num_particles=1000,
            num_types=6,
            dt=0.004,
        )
    
    @classmethod
    def large_world(cls):
        """Large world for maximum emergence."""
        return cls(
            world_size=200.0,
            num_particles=30000,
            num_types=24,
            dt=0.004,
            substeps=3,
        )
    
    @classmethod
    def huge_world(cls):
        """Massive world for full emergent dynamics."""
        return cls(
            world_size=500.0,
            num_particles=50000,
            num_types=32,
            dt=0.003,
            substeps=4,
            friction=0.6,
        )
    
    @classmethod
    def classic_emergence(cls):
        """
        CLASSIC MODE - Matches original Haskell particle-life dynamics.
        
        Key differences from default:
        1. LONG RANGE interactions (50% of world) - creates global structures
        2. HEAVY damping (Haskell uses 0.5 velocity multiplier per frame)
        3. Smaller particle count for clarity
        4. 4 classic species (Red, Green, Blue, Yellow)
        5. Simple linear force with bell-curve profile
        
        This mode prioritizes emergent complexity over raw physics.
        """
        return cls(
            world_size=2.0,  # Normalized world like Haskell [-1, 1]
            num_particles=500,  # Classic Haskell default
            num_types=4,  # Red, Green, Blue, Yellow
            dt=0.016,  # ~60fps timestep
            substeps=1,
            
            # CRITICAL: Long-range interactions (50% of world!)
            default_max_radius_frac=0.5,  # Attraction radius = half world
            default_min_radius_frac=0.15,  # Repulsion radius = 30% of max (Haskell: 0.3)
            
            # CRITICAL: Heavy damping like Haskell's * 0.5 per frame
            friction=30.0,  # Very high - halves velocity rapidly
            angular_friction=10.0,
            
            # Disable thermostat - let friction handle it
            thermostat_enabled=False,
            max_velocity=2.0,  # Reasonable for [-1,1] world
            
            # Disable wave mechanics for classic mode
            wave_repulsion_strength=0.0,
            
            # Smaller display for classic feel
            width=1080,
            height=1080,
        )
    
    @classmethod
    def emergence_advanced(cls):
        """
        ADVANCED EMERGENCE - Best of both worlds.
        
        Long-range forces like classic + advanced physics features.
        Tuned for visible emergent structures with wave/spin effects.
        """
        return cls(
            world_size=20.0,  # Moderate world
            num_particles=2000,  # Visible but not overwhelming
            num_types=8,  # More species for variety
            dt=0.008,
            substeps=2,
            
            # Long-range interactions (25% of world)
            default_max_radius_frac=0.25,
            default_min_radius_frac=0.05,
            
            # Moderate damping
            friction=5.0,
            angular_friction=3.0,
            
            # Gentle thermostat
            thermostat_enabled=True,
            target_temperature=0.3,
            thermostat_coupling=0.05,
            max_velocity=3.0,
            
            # Enable wave mechanics at moderate strength
            wave_repulsion_strength=10.0,
            wave_repulsion_exp=4.0,
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
