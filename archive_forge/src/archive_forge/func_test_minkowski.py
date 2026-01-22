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
def test_minkowski(self):
    for x, y in self.cases:
        dist1 = minkowski(x, y, p=1)
        assert_almost_equal(dist1, 3.0)
        dist1p5 = minkowski(x, y, p=1.5)
        assert_almost_equal(dist1p5, (1.0 + 2.0 ** 1.5) ** (2.0 / 3))
        dist2 = minkowski(x, y, p=2)
        assert_almost_equal(dist2, 5.0 ** 0.5)
        dist0p25 = minkowski(x, y, p=0.25)
        assert_almost_equal(dist0p25, (1.0 + 2.0 ** 0.25) ** 4)
    a = np.array([352, 916])
    b = np.array([350, 660])
    assert_equal(minkowski(a, b), minkowski(a.astype('uint16'), b.astype('uint16')))