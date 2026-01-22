import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import lsqr
def test_gh_2466():
    row = np.array([0, 0])
    col = np.array([0, 1])
    val = np.array([1, -1])
    A = scipy.sparse.coo_matrix((val, (row, col)), shape=(1, 2))
    b = np.asarray([4])
    lsqr(A, b)