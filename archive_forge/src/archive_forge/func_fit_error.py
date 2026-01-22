from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def fit_error(self, points, fit='least_square_distance'):
    """Evaluate the error for a list of points with respect to this plane.

        Args:
            points: List of points.
            fit: Type of fit error.

        Returns:
            Error for a list of points with respect to this plane.
        """
    if fit == 'least_square_distance':
        return self.fit_least_square_distance_error(points)
    if fit == 'maximum_distance':
        return self.fit_maximum_distance_error(points)
    return None