import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
@staticmethod
def from_extents(*args, minpos=None):
    """
        Create a new Bbox from *left*, *bottom*, *right* and *top*.

        The *y*-axis increases upwards.

        Parameters
        ----------
        left, bottom, right, top : float
            The four extents of the bounding box.
        minpos : float or None
            If this is supplied, the Bbox will have a minimum positive value
            set. This is useful when dealing with logarithmic scales and other
            scales where negative bounds result in floating point errors.
        """
    bbox = Bbox(np.reshape(args, (2, 2)))
    if minpos is not None:
        bbox._minpos[:] = minpos
    return bbox