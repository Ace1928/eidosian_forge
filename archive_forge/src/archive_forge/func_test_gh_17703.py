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
def test_gh_17703():
    arr_1 = np.array([1, 0, 0])
    arr_2 = np.array([2, 0, 0])
    expected = dice(arr_1, arr_2)
    actual = pdist([arr_1, arr_2], metric='dice')
    assert_allclose(actual, expected)
    actual = cdist(np.atleast_2d(arr_1), np.atleast_2d(arr_2), metric='dice')
    assert_allclose(actual, expected)