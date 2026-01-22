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
def set_masses(self, masses='defaults'):
    """Set atomic masses in atomic mass units.

        The array masses should contain a list of masses.  In case
        the masses argument is not given or for those elements of the
        masses list that are None, standard values are set."""
    if isinstance(masses, str):
        if masses == 'defaults':
            masses = atomic_masses[self.arrays['numbers']]
        elif masses == 'most_common':
            masses = atomic_masses_common[self.arrays['numbers']]
    elif masses is None:
        pass
    elif not isinstance(masses, np.ndarray):
        masses = list(masses)
        for i, mass in enumerate(masses):
            if mass is None:
                masses[i] = atomic_masses[self.numbers[i]]
    self.set_array('masses', masses, float, ())