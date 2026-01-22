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
def test_constructor1(self):
    indptr = array([0, 2, 2, 4])
    indices = array([0, 2, 2, 3])
    data = zeros((4, 2, 3))
    data[0] = array([[0, 1, 2], [3, 0, 5]])
    data[1] = array([[0, 2, 4], [6, 0, 10]])
    data[2] = array([[0, 4, 8], [12, 0, 20]])
    data[3] = array([[0, 5, 10], [15, 0, 25]])
    A = kron([[1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 4, 5]], [[0, 1, 2], [3, 0, 5]])
    Asp = bsr_matrix((data, indices, indptr), shape=(6, 12))
    assert_equal(Asp.toarray(), A)
    Asp = bsr_matrix((data, indices, indptr))
    assert_equal(Asp.toarray(), A)