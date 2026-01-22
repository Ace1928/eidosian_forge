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
def test_is_valid_dm_nonzero_diagonal_F(self):
    y = np.random.rand(10)
    D = squareform(y)
    for i in range(0, 5):
        D[i, i] = 2.0
    assert_equal(is_valid_dm(D), False)