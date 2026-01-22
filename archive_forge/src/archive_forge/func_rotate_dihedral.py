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
def rotate_dihedral(self, a1, a2, a3, a4, angle=None, mask=None, indices=None):
    """Rotate dihedral angle.

        Same usage as in :meth:`ase.Atoms.set_dihedral`: Rotate a group by a
        predefined dihedral angle, starting from its current configuration.
        """
    start = self.get_dihedral(a1, a2, a3, a4)
    self.set_dihedral(a1, a2, a3, a4, angle + start, mask, indices)