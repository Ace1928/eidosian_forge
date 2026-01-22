from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def line_intersection(self, point1: Sequence[float], point2: Sequence[float], tolerance: float=1e-08):
    """Computes the intersection points of a line with a simplex.

        Args:
            point1 (Sequence[float]): 1st point to determine the line.
            point2 (Sequence[float]): 2nd point to determine the line.
            tolerance (float): Tolerance for checking if an intersection is in the simplex. Defaults to 1e-8.

        Returns:
            points where the line intersects the simplex (0, 1, or 2).
        """
    b1 = self.bary_coords(point1)
    b2 = self.bary_coords(point2)
    line = b1 - b2
    valid = np.abs(line) > 1e-10
    possible = b1 - (b1[valid] / line[valid])[:, None] * line
    barys: list = []
    for p in possible:
        if (p >= -tolerance).all():
            found = False
            for b in barys:
                if np.all(np.abs(b - p) < tolerance):
                    found = True
                    break
            if not found:
                barys.append(p)
    assert len(barys) < 3, 'More than 2 intersections found'
    return [self.point_from_bary_coords(b) for b in barys]