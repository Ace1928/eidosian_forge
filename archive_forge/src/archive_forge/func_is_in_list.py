from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def is_in_list(self, plane_list) -> bool:
    """
        Checks whether the plane is identical to one of the Planes in the plane_list list of Planes

        Args:
            plane_list: List of Planes to be compared to

        Returns:
            bool: True if the plane is in the list.
        """
    return any((self.is_same_plane_as(plane) for plane in plane_list))