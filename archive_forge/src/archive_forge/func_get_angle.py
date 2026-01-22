from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def get_angle(v1: ArrayLike, v2: ArrayLike, units: Literal['degrees', 'radians']='degrees') -> float:
    """Calculates the angle between two vectors.

    Args:
        v1: Vector 1
        v2: Vector 2
        units: "degrees" or "radians". Defaults to "degrees".

    Returns:
        Angle between them in degrees.
    """
    d = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    d = min(d, 1)
    d = max(d, -1)
    angle = math.acos(d)
    if units == 'degrees':
        return math.degrees(angle)
    if units == 'radians':
        return angle
    raise ValueError(f'Invalid units={units!r}')