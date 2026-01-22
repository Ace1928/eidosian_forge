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
def test_cdist_mahalanobis(self):
    x1 = np.array([[2], [3]])
    x2 = np.array([[2], [5]])
    dist = cdist(x1, x2, metric='mahalanobis')
    assert_allclose(dist, [[0.0, np.sqrt(4.5)], [np.sqrt(0.5), np.sqrt(2)]])
    x1 = np.array([[0, 0], [-1, 0]])
    x2 = np.array([[0, 2], [1, 0], [0, -2]])
    dist = cdist(x1, x2, metric='mahalanobis')
    rt2 = np.sqrt(2)
    assert_allclose(dist, [[rt2, rt2, rt2], [2, 2 * rt2, 2]])
    with pytest.raises(ValueError):
        cdist([[0, 1]], [[2, 3]], metric='mahalanobis')