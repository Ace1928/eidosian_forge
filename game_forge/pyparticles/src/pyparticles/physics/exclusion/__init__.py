"""
Eidosian PyParticles V6 - Exclusion Mechanics Module

Implements Pauli-like exclusion and spin dynamics:
- Fermionic (identical particles repel)
- Bosonic (identical particles attract/condense)
- Spin quantum numbers (up/down)
- Spin-dependent interactions
"""

from .types import (
    SpinState,
    ParticleBehavior,
    ExclusionConfig,
    SpinConfig,
)
from .kernels import (
    compute_exclusion_force,
    compute_spin_interaction,
    apply_spin_flip,
    compute_spin_statistics,
)
from .registry import ExclusionRegistry

__all__ = [
    'SpinState',
    'ParticleBehavior', 
    'ExclusionConfig',
    'SpinConfig',
    'compute_exclusion_force',
    'compute_spin_interaction',
    'apply_spin_flip',
    'compute_spin_statistics',
    'ExclusionRegistry',
]
