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
def test_minmax_axis(self):
    D = np.arange(50).reshape(5, 10)
    D[1, :] = 0
    D[:, 9] = 0
    D[3, 3] = 0
    D[2, 2] = -1
    X = self.spcreator(D)
    axes = [-2, -1, 0, 1]
    for axis in axes:
        assert_array_equal(X.max(axis=axis).toarray(), D.max(axis=axis, keepdims=True))
        assert_array_equal(X.min(axis=axis).toarray(), D.min(axis=axis, keepdims=True))
    D = np.arange(1, 51).reshape(10, 5)
    X = self.spcreator(D)
    for axis in axes:
        assert_array_equal(X.max(axis=axis).toarray(), D.max(axis=axis, keepdims=True))
        assert_array_equal(X.min(axis=axis).toarray(), D.min(axis=axis, keepdims=True))
    D = np.zeros((10, 5))
    X = self.spcreator(D)
    for axis in axes:
        assert_array_equal(X.max(axis=axis).toarray(), D.max(axis=axis, keepdims=True))
        assert_array_equal(X.min(axis=axis).toarray(), D.min(axis=axis, keepdims=True))
    axes_even = [0, -2]
    axes_odd = [1, -1]
    D = np.zeros((0, 10))
    X = self.spcreator(D)
    for axis in axes_even:
        assert_raises(ValueError, X.min, axis=axis)
        assert_raises(ValueError, X.max, axis=axis)
    for axis in axes_odd:
        assert_array_equal(np.zeros((0, 1)), X.min(axis=axis).toarray())
        assert_array_equal(np.zeros((0, 1)), X.max(axis=axis).toarray())
    D = np.zeros((10, 0))
    X = self.spcreator(D)
    for axis in axes_odd:
        assert_raises(ValueError, X.min, axis=axis)
        assert_raises(ValueError, X.max, axis=axis)
    for axis in axes_even:
        assert_array_equal(np.zeros((1, 0)), X.min(axis=axis).toarray())
        assert_array_equal(np.zeros((1, 0)), X.max(axis=axis).toarray())