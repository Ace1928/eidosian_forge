from __future__ import annotations
import functools
import warnings
from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.json import MSONable
from scipy.constants import value as _cd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from pymatgen.core import Structure, get_el_sp
from pymatgen.core.spectrum import Spectrum
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.util.coord import get_linear_interpolated_value
def get_densities(self, spin: Spin | None=None):
    """Returns the density of states for a particular spin.

        Args:
            spin: Spin

        Returns:
            Returns the density of states for a particular spin. If Spin is
            None, the sum of all spins is returned.
        """
    if self.densities is None:
        result = None
    elif spin is None:
        if Spin.down in self.densities:
            result = self.densities[Spin.up] + self.densities[Spin.down]
        else:
            result = self.densities[Spin.up]
    else:
        result = self.densities[spin]
    return result