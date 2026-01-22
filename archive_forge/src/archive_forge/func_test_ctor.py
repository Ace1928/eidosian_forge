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
def test_ctor(self):
    assert_raises(TypeError, dok_matrix)
    b = array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 3]], 'd')
    A = dok_matrix(b)
    assert_equal(b.dtype, A.dtype)
    assert_equal(A.toarray(), b)
    c = csr_matrix(b)
    assert_equal(A.toarray(), c.toarray())
    data = [[0, 1, 2], [3, 0, 0]]
    d = dok_matrix(data, dtype=np.float32)
    assert_equal(d.dtype, np.float32)
    da = d.toarray()
    assert_equal(da.dtype, np.float32)
    assert_array_equal(da, data)