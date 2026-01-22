from __future__ import annotations
import bisect
from copy import copy, deepcopy
from datetime import datetime
from math import log, pi, sqrt
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MSONable
from scipy import constants
from scipy.special import comb, erfc
from pymatgen.core.structure import Structure
from pymatgen.util.due import Doi, due
@property
def total_energy_matrix(self):
    """
        The total energy matrix. Each matrix element (i, j) corresponds to the
        total interaction energy between site i and site j.

        Note that this does not include the charged-cell energy, which is only important
        when the simulation cell is not charge balanced.
        """
    if not self._initialized:
        self._calc_ewald_terms()
        self._initialized = True
    total_energy = self._recip + self._real
    for idx, energy in enumerate(self._point):
        total_energy[idx, idx] += energy
    return total_energy