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
def test_eliminate_zeros_all_zero(self):
    np.random.seed(0)
    m = bsr_matrix(np.random.random((12, 12)), blocksize=(2, 3))
    m.data[m.data <= 0.9] = 0
    m.eliminate_zeros()
    assert_equal(m.nnz, 66)
    assert_array_equal(m.data.shape, (11, 2, 3))
    m.data[m.data <= 1.0] = 0
    m.eliminate_zeros()
    assert_equal(m.nnz, 0)
    assert_array_equal(m.data.shape, (0, 2, 3))
    assert_array_equal(m.toarray(), np.zeros((12, 12)))
    m.eliminate_zeros()
    assert_equal(m.nnz, 0)
    assert_array_equal(m.data.shape, (0, 2, 3))
    assert_array_equal(m.toarray(), np.zeros((12, 12)))