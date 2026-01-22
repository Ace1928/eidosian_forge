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
@deprecated('Please use atoms.cell.reciprocal()')
def get_reciprocal_cell(self):
    """Get the three reciprocal lattice vectors as a 3x3 ndarray.

        Note that the commonly used factor of 2 pi for Fourier
        transforms is not included here."""
    return self.cell.reciprocal()