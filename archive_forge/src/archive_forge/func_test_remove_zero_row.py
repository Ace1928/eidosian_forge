import numpy as np
from numpy.testing import (
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix
def test_remove_zero_row(self):
    A = np.eye(3)
    A[1, :] = 0
    b = np.random.rand(3)
    b[1] = 0
    A1, b1, status, message = self.rr(A, b)
    assert_equal(status, 0)
    assert_allclose(A1, A[[0, 2], :])
    assert_allclose(b1, b[[0, 2]])