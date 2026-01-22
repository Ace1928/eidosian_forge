import copy
import numbers
from math import cos, sin, pi
import numpy as np
import ase.units as units
from ase.atom import Atom
from ase.cell import Cell
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.data import atomic_masses, atomic_masses_common
from ase.geometry import (wrap_positions, find_mic, get_angles, get_distances,
from ase.symbols import Symbols, symbols2numbers
from ase.utils import deprecated
def get_scaled_positions(self, wrap=True):
    """Get positions relative to unit cell.

        If wrap is True, atoms outside the unit cell will be wrapped into
        the cell in those directions with periodic boundary conditions
        so that the scaled coordinates are between zero and one.

        If any cell vectors are zero, the corresponding coordinates
        are evaluated as if the cell were completed using
        ``cell.complete()``.  This means coordinates will be Cartesian
        as long as the non-zero cell vectors span a Cartesian axis or
        plane."""
    fractional = self.cell.scaled_positions(self.positions)
    if wrap:
        for i, periodic in enumerate(self.pbc):
            if periodic:
                fractional[:, i] %= 1.0
                fractional[:, i] %= 1.0
    return fractional