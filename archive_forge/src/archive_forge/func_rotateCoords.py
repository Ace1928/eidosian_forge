from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def rotateCoords(coords, R):
    """
    Rotate the list of points using rotation matrix R

    Args:
        coords: List of points to be rotated
        R: Rotation matrix

    Returns:
        List of rotated points.
    """
    new_coords = []
    for pp in coords:
        rpp = matrixTimesVector(R, pp)
        new_coords.append(rpp)
    return new_coords