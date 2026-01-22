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
def test_correlation_positive(self):
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, 0.0, -2.0, 0.0, -2.0, 0.0, 0.0, -1.0, -2.0, 0.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, -2.0, 0.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0])
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 2.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0])
    dist = correlation(x, y)
    assert 0 <= dist <= 10 * np.finfo(np.float64).eps