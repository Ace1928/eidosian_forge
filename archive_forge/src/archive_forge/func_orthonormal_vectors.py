from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def orthonormal_vectors(self):
    """
        Returns a list of three orthogonal vectors, the two first being parallel to the plane and the
        third one is the normal vector of the plane

        Returns:
            List of orthogonal vectors
        :raise: ValueError if all the coefficients are zero or if there is some other strange error.
        """
    if self.e1 is None:
        diff = self.p2 - self.p1
        self.e1 = diff / norm(diff)
        self.e2 = np.cross(self.e3, self.e1)
    return [self.e1, self.e2, self.e3]