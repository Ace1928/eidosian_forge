from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def is_same_plane_as(self, plane) -> bool:
    """
        Checks whether the plane is identical to another Plane "plane"

        Args:
            plane: Plane to be compared to

        Returns:
            bool: True if the two facets are identical.
        """
    return np.allclose(self._coefficients, plane.coefficients)