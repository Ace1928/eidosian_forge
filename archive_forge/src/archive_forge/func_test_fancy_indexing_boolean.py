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
def test_fancy_indexing_boolean(self):
    np.random.seed(1234)
    B = asmatrix(arange(50).reshape(5, 10))
    A = self.spcreator(B)
    I = np.array(np.random.randint(0, 2, size=5), dtype=bool)
    J = np.array(np.random.randint(0, 2, size=10), dtype=bool)
    X = np.array(np.random.randint(0, 2, size=(5, 10)), dtype=bool)
    assert_equal(toarray(A[I]), B[I])
    assert_equal(toarray(A[:, J]), B[:, J])
    assert_equal(toarray(A[X]), B[X])
    assert_equal(toarray(A[B > 9]), B[B > 9])
    I = np.array([True, False, True, True, False])
    J = np.array([False, True, True, False, True, False, False, False, False, False])
    assert_equal(toarray(A[I, J]), B[I, J])
    Z1 = np.zeros((6, 11), dtype=bool)
    Z2 = np.zeros((6, 11), dtype=bool)
    Z2[0, -1] = True
    Z3 = np.zeros((6, 11), dtype=bool)
    Z3[-1, 0] = True
    assert_equal(A[Z1], np.array([]))
    assert_raises(IndexError, A.__getitem__, Z2)
    assert_raises(IndexError, A.__getitem__, Z3)
    assert_raises((IndexError, ValueError), A.__getitem__, (X, 1))