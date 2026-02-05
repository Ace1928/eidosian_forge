"""
Eidosian PyParticles V6 - Force Registry

Central registry for managing and configuring force interactions.
Supports enabling/disabling forces per species pair and dynamic force composition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np
from enum import IntEnum

from .base import ForceDefinition, DropoffType


class ForceType(IntEnum):
    """Force type identifiers matching the unified kernel."""
    LINEAR = 0
    INVERSE_SQUARE = 1
    INVERSE_CUBE = 2
    REPEL_ONLY = 3
    INVERSE = 4
    YUKAWA = 5
    LENNARD_JONES = 6
    MORSE = 7
    GAUSSIAN = 8
    EXPONENTIAL = 9


@dataclass
class ForceRegistry:
    """
    Central registry for all force definitions in the simulation.
    
    Manages:
    - Multiple force rules with different potentials
    - Per-species interaction matrices
    - Enable/disable per force and per species pair
    - Packing forces into arrays for Numba kernels
    """
    
    forces: List[ForceDefinition] = field(default_factory=list)
    num_types: int = 6
    _skip_defaults: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Initialize with default forces if empty."""
        if not self.forces and not self._skip_defaults:
            self._init_default_forces()
    
    def _init_default_forces(self):
        """Create default force configuration."""
        # Particle Life (linear attraction/repulsion)
        mat_linear = np.random.uniform(-1.0, 1.0, (self.num_types, self.num_types)).astype(np.float32)
        self.add_force(ForceDefinition(
            name="Particle Life",
            dropoff=DropoffType.LINEAR,
            matrix=mat_linear,
            max_radius=0.1,
            min_radius=0.02,
            strength=1.0,
            params=np.array([0.0], dtype=np.float32),  # No softening for linear
        ))
        
        # Gravity-like (inverse square, disabled by default)
        mat_grav = np.zeros((self.num_types, self.num_types), dtype=np.float32)
        self.add_force(ForceDefinition(
            name="Gravity",
            dropoff=DropoffType.INVERSE_SQUARE,
            matrix=mat_grav,
            max_radius=0.5,
            min_radius=0.01,
            strength=0.5,
            params=np.array([0.05], dtype=np.float32),  # softening
            enabled=False,
        ))
        
        # Strong nuclear-like (inverse cube)
        mat_strong = np.zeros((self.num_types, self.num_types), dtype=np.float32)
        self.add_force(ForceDefinition(
            name="Strong Force",
            dropoff=DropoffType.INVERSE_CUBE,
            matrix=mat_strong,
            max_radius=0.15,
            min_radius=0.01,
            strength=2.0,
            params=np.array([0.02], dtype=np.float32),
            enabled=False,
        ))
        
        # Lennard-Jones (molecular)
        mat_lj = np.zeros((self.num_types, self.num_types), dtype=np.float32)
        self.add_force(ForceDefinition(
            name="Lennard-Jones",
            dropoff=DropoffType.LENNARD_JONES,
            matrix=mat_lj,
            max_radius=0.3,
            min_radius=0.01,
            strength=1.0,
            params=np.array([0.005, 0.03], dtype=np.float32),  # softening, sigma
            enabled=False,
        ))
        
        # Yukawa (screened)
        mat_yukawa = np.zeros((self.num_types, self.num_types), dtype=np.float32)
        self.add_force(ForceDefinition(
            name="Yukawa",
            dropoff=DropoffType.YUKAWA,
            matrix=mat_yukawa,
            max_radius=0.4,
            min_radius=0.01,
            strength=1.0,
            params=np.array([0.01, 0.1], dtype=np.float32),  # softening, decay_length
            enabled=False,
        ))
        
        # Morse (bond-like)
        mat_morse = np.zeros((self.num_types, self.num_types), dtype=np.float32)
        self.add_force(ForceDefinition(
            name="Morse Bond",
            dropoff=DropoffType.MORSE,
            matrix=mat_morse,
            max_radius=0.3,
            min_radius=0.01,
            strength=1.0,
            params=np.array([0.01, 0.08, 10.0], dtype=np.float32),  # softening, r0, well_width
            enabled=False,
        ))
    
    def add_force(self, force: ForceDefinition) -> int:
        """Add a force definition and return its index."""
        self.forces.append(force)
        return len(self.forces) - 1
    
    def remove_force(self, index: int) -> Optional[ForceDefinition]:
        """Remove force at index and return it."""
        if 0 <= index < len(self.forces):
            return self.forces.pop(index)
        return None
    
    def get_force(self, name: str) -> Optional[ForceDefinition]:
        """Get force by name."""
        for f in self.forces:
            if f.name == name:
                return f
        return None
    
    def enable_force(self, name: str, enabled: bool = True):
        """Enable or disable a force by name."""
        f = self.get_force(name)
        if f:
            f.enabled = enabled
    
    def get_active_forces(self) -> List[ForceDefinition]:
        """Return list of enabled forces."""
        return [f for f in self.forces if f.enabled]
    
    def set_num_types(self, n_types: int):
        """Resize all matrices for new number of species."""
        if n_types == self.num_types:
            return
        
        old_n = self.num_types
        self.num_types = n_types
        
        for force in self.forces:
            old_matrix = force.matrix
            new_matrix = np.zeros((n_types, n_types), dtype=np.float32)
            
            # Copy existing values where possible
            min_n = min(old_n, n_types)
            new_matrix[:min_n, :min_n] = old_matrix[:min_n, :min_n]
            
            # Randomize new species interactions
            if n_types > old_n:
                new_matrix[old_n:, :] = np.random.uniform(-1, 1, (n_types - old_n, n_types))
                new_matrix[:, old_n:] = np.random.uniform(-1, 1, (n_types, n_types - old_n))
            
            force.matrix = new_matrix
    
    def pack_for_kernel(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pack active forces into arrays for Numba kernel.
        
        Returns:
            matrices: (N_forces, T, T) interaction matrices
            params: (N_forces, 8) force parameters
                    [min_r, max_r, strength, softening, force_type, p1, p2, p3]
            extra_params: (N_forces, 4) extra parameters per force
        """
        active = self.get_active_forces()
        n_forces = len(active)
        n_types = self.num_types
        
        if n_forces == 0:
            # Return empty arrays with correct shape
            return (
                np.zeros((1, n_types, n_types), dtype=np.float32),
                np.zeros((1, 8), dtype=np.float32),
                np.zeros((1, 4), dtype=np.float32),
            )
        
        matrices = np.zeros((n_forces, n_types, n_types), dtype=np.float32)
        params = np.zeros((n_forces, 8), dtype=np.float32)
        extra_params = np.zeros((n_forces, 4), dtype=np.float32)
        
        for i, force in enumerate(active):
            matrices[i] = force.matrix
            params[i, 0] = force.min_radius
            params[i, 1] = force.max_radius
            params[i, 2] = force.strength
            params[i, 3] = force.params[0] if len(force.params) > 0 else 0.01  # softening
            params[i, 4] = float(self._dropoff_to_forcetype(force.dropoff))
            
            # Pack extra params
            for j, p in enumerate(force.params[1:]):
                if j < 3:
                    extra_params[i, j] = p
        
        return matrices, params, extra_params
    
    def _dropoff_to_forcetype(self, dropoff: DropoffType) -> int:
        """Convert DropoffType to kernel ForceType integer."""
        mapping = {
            DropoffType.LINEAR: 0,
            DropoffType.INVERSE_SQUARE: 1,
            DropoffType.INVERSE_CUBE: 2,
            DropoffType.INVERSE: 4,
            DropoffType.YUKAWA: 5,
            DropoffType.LENNARD_JONES: 6,
            DropoffType.MORSE: 7,
            DropoffType.GAUSSIAN: 8,
            DropoffType.EXPONENTIAL: 9,
        }
        return mapping.get(dropoff, 0)
    
    def randomize_all(self, low: float = -1.0, high: float = 1.0):
        """Randomize all force matrices."""
        for force in self.forces:
            force.randomize_matrix(low, high)
    
    def clear_all(self):
        """Set all matrices to zero."""
        for force in self.forces:
            force.matrix.fill(0.0)
    
    def get_max_radius(self) -> float:
        """Get maximum interaction radius across all enabled forces."""
        active = self.get_active_forces()
        if not active:
            return 0.1
        return max(f.max_radius for f in active)
    
    def to_dict(self) -> dict:
        """Serialize registry to dictionary."""
        return {
            'num_types': self.num_types,
            'forces': [
                {
                    'name': f.name,
                    'dropoff': f.dropoff.value,
                    'matrix': f.matrix.tolist(),
                    'max_radius': f.max_radius,
                    'min_radius': f.min_radius,
                    'strength': f.strength,
                    'params': f.params.tolist(),
                    'enabled': f.enabled,
                    'symmetric': f.symmetric,
                }
                for f in self.forces
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ForceRegistry':
        """Deserialize registry from dictionary."""
        registry = cls(num_types=data['num_types'], forces=[], _skip_defaults=True)
        
        for fd in data['forces']:
            registry.add_force(ForceDefinition(
                name=fd['name'],
                dropoff=DropoffType(fd['dropoff']),
                matrix=np.array(fd['matrix'], dtype=np.float32),
                max_radius=fd['max_radius'],
                min_radius=fd['min_radius'],
                strength=fd['strength'],
                params=np.array(fd['params'], dtype=np.float32),
                enabled=fd['enabled'],
                symmetric=fd.get('symmetric', True),
            ))
        
        return registry


def create_preset_registry(preset: str, num_types: int = 6) -> ForceRegistry:
    """
    Create a force registry with preset configurations.
    
    Presets:
    - 'particle_life': Classic particle life
    - 'molecular': LJ + Morse for molecular dynamics
    - 'plasma': Yukawa for plasma-like behavior
    - 'gravitational': Inverse square for gravity
    - 'crystal': Strong short-range + weak long-range
    """
    registry = ForceRegistry(num_types=num_types, forces=[], _skip_defaults=True)
    
    if preset == 'particle_life':
        mat = np.random.uniform(-1, 1, (num_types, num_types)).astype(np.float32)
        registry.add_force(ForceDefinition(
            name="Particle Life",
            dropoff=DropoffType.LINEAR,
            matrix=mat,
            max_radius=0.12,
            min_radius=0.02,
            strength=1.0,
        ))
        
    elif preset == 'molecular':
        # Lennard-Jones for all types
        mat_lj = np.full((num_types, num_types), 1.0, dtype=np.float32)
        registry.add_force(ForceDefinition(
            name="LJ",
            dropoff=DropoffType.LENNARD_JONES,
            matrix=mat_lj,
            max_radius=0.25,
            strength=0.5,
            params=np.array([0.005, 0.04], dtype=np.float32),
        ))
        
    elif preset == 'plasma':
        # Yukawa with type-dependent charges
        mat = np.ones((num_types, num_types), dtype=np.float32)
        for i in range(num_types):
            mat[i, i] = -1.0  # Same type repels
        registry.add_force(ForceDefinition(
            name="Screened Coulomb",
            dropoff=DropoffType.YUKAWA,
            matrix=mat,
            max_radius=0.5,
            strength=2.0,
            params=np.array([0.01, 0.15], dtype=np.float32),
        ))
        
    elif preset == 'gravitational':
        mat = np.ones((num_types, num_types), dtype=np.float32)
        registry.add_force(ForceDefinition(
            name="Gravity",
            dropoff=DropoffType.INVERSE_SQUARE,
            matrix=mat,
            max_radius=1.0,
            strength=0.1,
            params=np.array([0.02], dtype=np.float32),
        ))
        
    elif preset == 'crystal':
        # Strong short-range repulsion
        mat_rep = np.full((num_types, num_types), -2.0, dtype=np.float32)
        registry.add_force(ForceDefinition(
            name="Core Repulsion",
            dropoff=DropoffType.INVERSE_CUBE,
            matrix=mat_rep,
            max_radius=0.08,
            strength=3.0,
            params=np.array([0.01], dtype=np.float32),
        ))
        # Weak long-range attraction
        mat_att = np.full((num_types, num_types), 0.5, dtype=np.float32)
        registry.add_force(ForceDefinition(
            name="Long-range Attraction",
            dropoff=DropoffType.EXPONENTIAL,
            matrix=mat_att,
            max_radius=0.4,
            strength=0.3,
            params=np.array([0.01, 0.15], dtype=np.float32),
        ))
    
    else:
        # Default: classic particle life
        return create_preset_registry('particle_life', num_types)
    
    return registry
