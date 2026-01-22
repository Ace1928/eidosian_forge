from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def init_3points(self, non_zeros, zeros):
    """Initialize three random points on this plane.

        Args:
            non_zeros: Indices of plane coefficients ([a, b, c]) that are not zero.
            zeros: Indices of plane coefficients ([a, b, c]) that are equal to zero.
        """
    if len(non_zeros) == 3:
        self.p1 = np.array([-self.d / self.a, 0.0, 0.0], float)
        self.p2 = np.array([0.0, -self.d / self.b, 0.0], float)
        self.p3 = np.array([0.0, 0.0, -self.d / self.c], float)
    elif len(non_zeros) == 2:
        self.p1 = np.zeros(3, float)
        self.p1[non_zeros[1]] = -self.d / self.coefficients[non_zeros[1]]
        self.p2 = np.array(self.p1)
        self.p2[zeros[0]] = 1.0
        self.p3 = np.zeros(3, float)
        self.p3[non_zeros[0]] = -self.d / self.coefficients[non_zeros[0]]
    elif len(non_zeros) == 1:
        self.p1 = np.zeros(3, float)
        self.p1[non_zeros[0]] = -self.d / self.coefficients[non_zeros[0]]
        self.p2 = np.array(self.p1)
        self.p2[zeros[0]] = 1.0
        self.p3 = np.array(self.p1)
        self.p3[zeros[1]] = 1.0