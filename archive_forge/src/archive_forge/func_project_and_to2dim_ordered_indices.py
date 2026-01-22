from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def project_and_to2dim_ordered_indices(self, pps, plane_center='mean'):
    """
        Projects each points in the point list pps on plane and returns the indices that would sort the
        list of projected points in anticlockwise order

        Args:
            pps: List of points to project on plane

        Returns:
            List of indices that would sort the list of projected points.
        """
    pp2d = self.project_and_to2dim(pps, plane_center)
    return anticlockwise_sort_indices(pp2d)