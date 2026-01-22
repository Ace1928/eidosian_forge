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
def test_eliminate_zeros(self):
    data = kron([1, 0, 0, 0, 2, 0, 3, 0], [[1, 1], [1, 1]]).T
    data = data.reshape(-1, 2, 2)
    indices = array([1, 2, 3, 4, 5, 6, 7, 8])
    indptr = array([0, 3, 8])
    asp = bsr_matrix((data, indices, indptr), shape=(4, 20))
    bsp = asp.copy()
    asp.eliminate_zeros()
    assert_array_equal(asp.nnz, 3 * 4)
    assert_array_equal(asp.toarray(), bsp.toarray())