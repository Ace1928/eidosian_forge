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
def reciprocal_space_energy(self):
    """The reciprocal space energy."""
    if not self._initialized:
        self._calc_ewald_terms()
        self._initialized = True
    return sum(sum(self._recip))