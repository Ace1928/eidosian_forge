import numpy as np
from numpy.testing import (
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix
def test_m_lt_n_sparse(self):
    np.random.seed(2017)
    m, n = (20, 50)
    p = 0.05
    A = np.random.rand(m, n)
    A[np.random.rand(m, n) > p] = 0
    rank = np.linalg.matrix_rank(A)
    b = np.zeros(A.shape[0])
    A1, b1, status, message = self.rr(A, b)
    assert_equal(status, 0)
    assert_equal(A1.shape[0], rank)
    assert_equal(np.linalg.matrix_rank(A1), rank)