from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def indices_separate(self, points, dist_tolerance):
    """
        Returns three lists containing the indices of the points lying on one side of the plane, on the plane
        and on the other side of the plane. The dist_tolerance parameter controls the tolerance to which a point
        is considered to lie on the plane or not (distance to the plane)

        Args:
            points: list of points
            dist_tolerance: tolerance to which a point is considered to lie on the plane
                or not (distance to the plane)

        Returns:
            The lists of indices of the points on one side of the plane, on the plane and
            on the other side of the plane.
        """
    side1 = []
    inplane = []
    side2 = []
    for ip, pp in enumerate(points):
        if self.is_in_plane(pp, dist_tolerance):
            inplane.append(ip)
        elif np.dot(pp + self.vector_to_origin, self.normal_vector) < 0.0:
            side1.append(ip)
        else:
            side2.append(ip)
    return [side1, inplane, side2]