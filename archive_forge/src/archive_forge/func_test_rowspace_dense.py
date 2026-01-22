import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import (TestCase, assert_array_almost_equal,
def test_rowspace_dense(self):
    A = np.array([[1, 2, 3, 4, 0, 5, 0, 7], [0, 8, 7, 0, 1, 5, 9, 0], [1, 0, 0, 0, 0, 1, 2, 3]])
    test_points = ([1, 2, 3], [1, 10, 3], [1.12, 10, 0])
    for method in available_dense_methods:
        _, _, Y = projections(A, method)
        for z in test_points:
            x = Y.matvec(z)
            assert_array_almost_equal(A.dot(x), z)
            A_ext = np.vstack((A, x))
            assert_equal(np.linalg.matrix_rank(A), np.linalg.matrix_rank(A_ext))