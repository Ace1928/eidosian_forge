import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import (TestCase, assert_array_almost_equal,
def test_compare_dense_and_sparse2(self):
    D1 = np.diag([-1.7, 1, 0.5])
    D2 = np.diag([1, -0.6, -0.3])
    D3 = np.diag([-0.3, -1.5, 2])
    A = np.hstack([D1, D2, D3])
    A_sparse = csc_matrix(A)
    np.random.seed(0)
    Z, LS, Y = projections(A)
    Z_sparse, LS_sparse, Y_sparse = projections(A_sparse)
    for k in range(1):
        z = np.random.normal(size=(9,))
        assert_array_almost_equal(Z.dot(z), Z_sparse.dot(z))
        assert_array_almost_equal(LS.dot(z), LS_sparse.dot(z))
        x = np.random.normal(size=(3,))
        assert_array_almost_equal(Y.dot(x), Y_sparse.dot(x))