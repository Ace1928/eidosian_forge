import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
def get_coeffs(self):
    """ Return spline coefficients."""
    return self.tck[2]