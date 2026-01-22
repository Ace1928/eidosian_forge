from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def matrixTimesVector(MM, aa):
    """

    Args:
        MM: A matrix of size 3x3
        aa: A vector of size 3

    Returns:
        A vector of size 3 which is the product of the matrix by the vector
    """
    bb = np.zeros(3, float)
    for ii in range(3):
        bb[ii] = np.sum(MM[ii, :] * aa)
    return bb