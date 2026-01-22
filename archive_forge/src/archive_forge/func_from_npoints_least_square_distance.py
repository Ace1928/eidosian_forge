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
def from_npoints_least_square_distance(cls, points) -> Self:
    """Initializes plane from a list of points using a least square fitting procedure.

        Args:
            points: List of points.

        Returns:
            Plane.
        """
    mean_point = np.array([sum((pp[ii] for pp in points)) for ii in range(3)], float)
    mean_point /= len(points)
    AA = np.zeros((len(points), 3), float)
    for ii, pp in enumerate(points):
        for jj in range(3):
            AA[ii, jj] = pp[jj] - mean_point[jj]
    _UU, SS, Vt = np.linalg.svd(AA)
    imin = np.argmin(SS)
    normal_vector = Vt[imin]
    non_zeros = np.argwhere(normal_vector != 0.0)
    if normal_vector[non_zeros[0, 0]] < 0.0:
        normal_vector = -normal_vector
    dd = -np.dot(normal_vector, mean_point)
    coefficients = np.array([normal_vector[0], normal_vector[1], normal_vector[2], dd], float)
    return cls(coefficients)