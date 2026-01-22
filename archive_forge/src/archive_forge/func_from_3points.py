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
def from_3points(cls, p1, p2, p3) -> Self:
    """Initializes plane from three points.

        Args:
            p1: First point.
            p2: Second point.
            p3: Third point.

        Returns:
            Plane.
        """
    nn = np.cross(p1 - p3, p2 - p3)
    normal_vector = nn / norm(nn)
    non_zeros = np.argwhere(normal_vector != 0.0)
    if normal_vector[non_zeros[0, 0]] < 0.0:
        normal_vector = -normal_vector
    dd = -np.dot(normal_vector, p1)
    coefficients = np.array([normal_vector[0], normal_vector[1], normal_vector[2], dd], float)
    return cls(coefficients, p1=p1, p2=p2, p3=p3)