import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def check_squareform_vector(self, dtype):
    v = np.zeros((0,), dtype=dtype)
    rv = squareform(v)
    assert_equal(rv.shape, (1, 1))
    assert_equal(rv.dtype, dtype)
    assert_array_equal(rv, [[0]])
    v = np.array([8.3], dtype=dtype)
    rv = squareform(v)
    assert_equal(rv.shape, (2, 2))
    assert_equal(rv.dtype, dtype)
    assert_array_equal(rv, np.array([[0, 8.3], [8.3, 0]], dtype=dtype))