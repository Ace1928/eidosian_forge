from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def fit_maximum_distance_error(self, points):
    """Evaluate the max distance error for a list of points with respect to this plane.

        Args:
            points: List of points.

        Returns:
            Max distance error for a list of points with respect to this plane.
        """
    return np.max([self.distance_to_point(pp) for pp in points])