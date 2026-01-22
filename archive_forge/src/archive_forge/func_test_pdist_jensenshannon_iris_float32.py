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
def test_pdist_jensenshannon_iris_float32(self):
    eps = 1e-06
    X = np.float32(eo['iris'])
    Y_right = eo['pdist-jensenshannon-iris']
    Y_test1 = pdist(X, 'jensenshannon')
    assert_allclose(Y_test1, Y_right, atol=eps, verbose=verbose > 2)