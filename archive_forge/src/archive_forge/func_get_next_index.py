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
@classmethod
def get_next_index(cls, matrix, manipulation, indices_left):
    """
        Returns an index that should have the most negative effect on the
        matrix sum.
        """
    f = manipulation[0]
    indices = list(indices_left.intersection(manipulation[2]))
    sums = np.sum(matrix[indices], axis=1)
    return indices[sums.argmax(axis=0)] if f < 1 else indices[sums.argmin(axis=0)]