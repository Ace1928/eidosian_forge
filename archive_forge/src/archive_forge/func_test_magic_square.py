import numpy as np
from numpy.testing import (
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix
def test_magic_square(self):
    A, b, c, numbers, _ = magic_square(3)
    A1, b1, status, message = self.rr(A, b)
    assert_equal(status, 0)
    assert_equal(A1.shape[0], 23)
    assert_equal(np.linalg.matrix_rank(A1), 23)