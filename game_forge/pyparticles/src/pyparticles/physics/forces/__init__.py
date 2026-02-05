"""
Eidosian PyParticles V6 - Force System

Modular, pluggable force implementations with various interaction potentials
and distance falloff functions.

Force Types:
- Linear: Classic particle-life attraction/repulsion
- Inverse: 1/r, 1/r², 1/r³ potentials
- Yukawa: Screened Coulomb (exponential decay)
- Lennard-Jones: Molecular dynamics 6-12 potential
- Morse: Bond-like interactions
- Gaussian: Localized soft forces
"""

from .base import (
    ForceKernel, 
    DropoffType, 
    ForceDefinition,
    DROPOFF_FUNCTIONS,
)
from .registry import ForceRegistry
from .potentials import (
    linear_force,
    inverse_force,
    inverse_square_force,
    inverse_cube_force,
    yukawa_force,
    lennard_jones_force,
    morse_force,
    gaussian_force,
)

__all__ = [
    'ForceKernel',
    'DropoffType',
    'ForceDefinition',
    'ForceRegistry',
    'DROPOFF_FUNCTIONS',
    'linear_force',
    'inverse_force',
    'inverse_square_force',
    'inverse_cube_force',
    'yukawa_force',
    'lennard_jones_force',
    'morse_force',
    'gaussian_force',
]
