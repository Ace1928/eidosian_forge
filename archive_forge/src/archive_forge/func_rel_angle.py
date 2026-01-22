from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@njit
def rel_angle(vec_set1, vec_set2):
    """
    Calculate the relative angle between two vector sets.

    Args:
        vec_set1(array[array]): an array of two vectors
        vec_set2(array[array]): second array of two vectors
    """
    return vec_angle(vec_set2[0], vec_set2[1]) / vec_angle(vec_set1[0], vec_set1[1]) - 1