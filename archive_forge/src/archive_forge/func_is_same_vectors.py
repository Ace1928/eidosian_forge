from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@njit
def is_same_vectors(vec_set1, vec_set2, bidirectional=False, max_length_tol=0.03, max_angle_tol=0.01) -> bool:
    """
    Determine if two sets of vectors are the same within length and angle
    tolerances
    Args:
        vec_set1(array[array]): an array of two vectors
        vec_set2(array[array]): second array of two vectors.
    """
    if bidirectional:
        return _bidirectional_same_vectors(vec_set1, vec_set2, max_length_tol=max_length_tol, max_angle_tol=max_angle_tol)
    return _unidirectional_is_same_vectors(vec_set1, vec_set2, max_length_tol=max_length_tol, max_angle_tol=max_angle_tol)