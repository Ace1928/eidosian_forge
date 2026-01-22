from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def coord_list_mapping_pbc(subset, superset, atol: float=1e-08, pbc: PbcLike=(True, True, True)):
    """Gives the index mapping from a subset to a superset.
    Superset cannot contain duplicate matching rows.

    Args:
        subset (ArrayLike): List of frac_coords
        superset (ArrayLike): List of frac_coords
        atol (float): Absolute tolerance. Defaults to 1e-8.
        pbc (tuple): A tuple defining the periodic boundary conditions along the three
            axis of the lattice.

    Returns:
        list of indices such that superset[indices] = subset
    """
    atol = np.ones(3) * atol
    return coord_cython.coord_list_mapping_pbc(subset, superset, atol, pbc)