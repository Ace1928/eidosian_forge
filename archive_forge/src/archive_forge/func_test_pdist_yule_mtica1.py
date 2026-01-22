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
def test_pdist_yule_mtica1(self):
    m = wyule(np.array([1, 0, 1, 1, 0]), np.array([1, 1, 0, 1, 1]))
    m2 = wyule(np.array([1, 0, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 1, 1], dtype=bool))
    if verbose > 2:
        print(m)
    assert_allclose(m, 2, rtol=0, atol=1e-10)
    assert_allclose(m2, 2, rtol=0, atol=1e-10)