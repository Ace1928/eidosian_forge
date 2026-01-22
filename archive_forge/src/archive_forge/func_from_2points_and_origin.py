from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
@classmethod
def from_2points_and_origin(cls, p1, p2) -> Self:
    """Initializes plane from two points and the origin.

        Args:
            p1: First point.
            p2: Second point.

        Returns:
            Plane.
        """
    return cls.from_3points(p1, p2, np.zeros(3))