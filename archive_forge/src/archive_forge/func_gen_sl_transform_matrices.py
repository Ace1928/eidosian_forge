from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@njit
def gen_sl_transform_matrices(area_multiple):
    """
    Generates the transformation matrices that convert a set of 2D
    vectors into a super lattice of integer area multiple as proven
    in Cassels:

    Cassels, John William Scott. An introduction to the geometry of
    numbers. Springer Science & Business Media, 2012.

    Args:
        area_multiple(int): integer multiple of unit cell area for super
        lattice area

    Returns:
        matrix_list: transformation matrices to convert unit vectors to
        super lattice vectors
    """
    return [np.array(((i, j), (0, area_multiple / i))) for i in get_factors(area_multiple) for j in range(area_multiple // i)]