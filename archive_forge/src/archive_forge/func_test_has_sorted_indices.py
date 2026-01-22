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
def test_has_sorted_indices(self):
    """Ensure has_sorted_indices memoizes sorted state for sort_indices"""
    sorted_inds = np.array([0, 1])
    unsorted_inds = np.array([1, 0])
    data = np.array([1, 1])
    indptr = np.array([0, 2])
    M = csr_matrix((data, sorted_inds, indptr)).copy()
    assert_equal(True, M.has_sorted_indices)
    assert isinstance(M.has_sorted_indices, bool)
    M = csr_matrix((data, unsorted_inds, indptr)).copy()
    assert_equal(False, M.has_sorted_indices)
    M.sort_indices()
    assert_equal(True, M.has_sorted_indices)
    assert_array_equal(M.indices, sorted_inds)
    M = csr_matrix((data, unsorted_inds, indptr)).copy()
    M.has_sorted_indices = True
    assert_equal(True, M.has_sorted_indices)
    assert_array_equal(M.indices, unsorted_inds)
    M.sort_indices()
    assert_array_equal(M.indices, unsorted_inds)