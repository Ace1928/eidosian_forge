from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def bary_coords(self, point):
    """
        Args:
            point (ArrayLike): Point coordinates.

        Returns:
            Barycentric coordinations.
        """
    try:
        return np.dot(np.concatenate([point, [1]]), self._aug_inv)
    except AttributeError as exc:
        raise ValueError('Simplex is not full-dimensional') from exc