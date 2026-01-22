from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def projectionpoints(self, pps):
    """
        Projects each points in the point list pps on plane and returns the list of projected points

        Args:
            pps: List of points to project on plane

        Returns:
            List of projected point on plane.
        """
    return [pp - np.dot(pp - self.p1, self.normal_vector) * self.normal_vector for pp in pps]