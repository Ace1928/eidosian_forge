import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def test_minmax(self):
    for dtype in [np.float32, np.float64, np.int32, np.int64, np.complex128]:
        D = np.arange(20, dtype=dtype).reshape(5, 4)
        X = self.spcreator(D)
        assert_equal(X.min(), 0)
        assert_equal(X.max(), 19)
        assert_equal(X.min().dtype, dtype)
        assert_equal(X.max().dtype, dtype)
        D *= -1
        X = self.spcreator(D)
        assert_equal(X.min(), -19)
        assert_equal(X.max(), 0)
        D += 5
        X = self.spcreator(D)
        assert_equal(X.min(), -14)
        assert_equal(X.max(), 5)
    X = self.spcreator(np.arange(1, 10).reshape(3, 3))
    assert_equal(X.min(), 1)
    assert_equal(X.min().dtype, X.dtype)
    X = -X
    assert_equal(X.max(), -1)
    Z = self.spcreator(np.zeros(1))
    assert_equal(Z.min(), 0)
    assert_equal(Z.max(), 0)
    assert_equal(Z.max().dtype, Z.dtype)
    D = np.arange(20, dtype=float).reshape(5, 4)
    D[0:2, :] = 0
    X = self.spcreator(D)
    assert_equal(X.min(), 0)
    assert_equal(X.max(), 19)
    for D in [np.zeros((0, 0)), np.zeros((0, 10)), np.zeros((10, 0))]:
        X = self.spcreator(D)
        assert_raises(ValueError, X.min)
        assert_raises(ValueError, X.max)