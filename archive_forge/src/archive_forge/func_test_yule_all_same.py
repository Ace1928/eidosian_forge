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
def test_yule_all_same():
    x = np.ones((2, 6), dtype=bool)
    d = wyule(x[0], x[0])
    assert d == 0.0
    d = pdist(x, 'yule')
    assert_equal(d, [0.0])
    d = cdist(x[:1], x[:1], 'yule')
    assert_equal(d, [[0.0]])