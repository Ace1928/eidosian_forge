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
@pytest.mark.parametrize('p', [0.1, 0.25, 1.0, 1.23, 2.0, 3.8, 4.6, np.inf])
def test_cdist_minkowski_random(self, p):
    eps = 1e-13
    X1 = eo['cdist-X1']
    X2 = eo['cdist-X2']
    Y1 = wcdist_no_const(X1, X2, 'minkowski', p=p)
    Y2 = wcdist_no_const(X1, X2, 'test_minkowski', p=p)
    assert_allclose(Y1, Y2, atol=0, rtol=eps, verbose=verbose > 2)