from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def in_coord_list(coord_list, coord, atol: float=1e-08) -> bool:
    """Tests if a particular coord is within a coord_list.

    Args:
        coord_list: List of coords to test
        coord: Specific coordinates
        atol: Absolute tolerance. Defaults to 1e-8. Accepts both scalar and
            array.

    Returns:
        bool: True if coord is in the coord list.
    """
    return len(find_in_coord_list(coord_list, coord, atol=atol)) > 0