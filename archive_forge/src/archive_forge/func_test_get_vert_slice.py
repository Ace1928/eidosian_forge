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
def test_get_vert_slice(self):
    B = arange(50.0).reshape(5, 10)
    A = self.spcreator(B)
    assert_array_equal(B[2:5, [0]], A[2:5, 0].toarray())
    assert_array_equal(B[:, [1]], A[:, 1].toarray())
    C = array([[1, 2, 1], [4, 0, 6], [0, 0, 0], [0, 0, 1]])
    D = self.spcreator(C)
    assert_array_equal(C[1:3, [1]], D[1:3, 1].toarray())
    assert_array_equal(C[:, [2]], D[:, 2].toarray())
    E = array([[1, 0, 1], [4, 0, 0], [0, 0, 0], [0, 0, 1]])
    F = self.spcreator(E)
    assert_array_equal(E[:, [1]], F[:, 1].toarray())
    assert_array_equal(E[-2:, [2]], F[-2:, 2].toarray())
    assert_raises(IndexError, A.__getitem__, (slice(None), 11))
    assert_raises(IndexError, A.__getitem__, (6, slice(3, 7)))