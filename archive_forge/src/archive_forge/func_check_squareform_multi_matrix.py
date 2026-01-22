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
def check_squareform_multi_matrix(self, n):
    X = np.random.rand(n, 4)
    Y = wpdist_no_const(X)
    assert_equal(len(Y.shape), 1)
    A = squareform(Y)
    Yr = squareform(A)
    s = A.shape
    k = 0
    if verbose >= 3:
        print(A.shape, Y.shape, Yr.shape)
    assert_equal(len(s), 2)
    assert_equal(len(Yr.shape), 1)
    assert_equal(s[0], s[1])
    for i in range(0, s[0]):
        for j in range(i + 1, s[1]):
            if i != j:
                assert_equal(A[i, j], Y[k])
                k += 1
            else:
                assert_equal(A[i, j], 0)