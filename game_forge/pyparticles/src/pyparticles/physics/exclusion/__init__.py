"""
Eidosian PyParticles V6.2 - Exclusion Mechanics Module

Implements Pauli-like exclusion and spin dynamics with WAVE PERIMETER integration:
- Fermionic (identical particles repel, enhanced by wave crests)
- Bosonic (identical particles attract/condense)
- Spin quantum numbers (up/down)
- Spin-dependent interactions
- Wave-deformed exclusion zones
- Angular velocity coupling
"""

from .types import (
    SpinState,
    ParticleBehavior,
    ExclusionConfig,
    SpinConfig,
)
from .kernels import (
    compute_exclusion_force_wave,
    wave_radius_at_angle,
    apply_exclusion_forces_wave,
    apply_spin_coupling,
    compute_spin_coupling_torque,
    apply_spin_flip,
    compute_spin_statistics,
    initialize_spins,
)
from .registry import ExclusionRegistry

__all__ = [
    'SpinState',
    'ParticleBehavior', 
    'ExclusionConfig',
    'SpinConfig',
    'compute_exclusion_force_wave',
    'wave_radius_at_angle',
    'apply_exclusion_forces_wave',
    'apply_spin_coupling',
    'compute_spin_coupling_torque',
    'apply_spin_flip',
    'compute_spin_statistics',
    'initialize_spins',
    'ExclusionRegistry',
]
