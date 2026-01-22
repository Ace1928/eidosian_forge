from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@njit
def reduce_vectors(a, b):
    """
    Generate independent and unique basis vectors based on the
    methodology of Zur and McGill.
    """
    if np.dot(a, b) < 0:
        return reduce_vectors(a, -b)
    fast_norm_b = fast_norm(b)
    if fast_norm(a) > fast_norm_b:
        return reduce_vectors(b, a)
    if fast_norm_b > fast_norm(np.add(b, a)):
        return reduce_vectors(a, np.add(b, a))
    if fast_norm_b > fast_norm(np.subtract(b, a)):
        return reduce_vectors(a, np.subtract(b, a))
    return (a, b)