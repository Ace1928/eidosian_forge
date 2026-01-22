import operator
import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename
from cupyx.scipy.sparse import csr_matrix
import numpy as np
@classmethod
def construct_fast(cls, t, c, k, extrapolate=True, axis=0):
    """Construct a spline without making checks.
        Accepts same parameters as the regular constructor. Input arrays
        `t` and `c` must of correct shape and dtype.
        """
    self = object.__new__(cls)
    self.t, self.c, self.k = (t, c, k)
    self.extrapolate = extrapolate
    self.axis = axis
    return self