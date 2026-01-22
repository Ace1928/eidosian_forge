import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import (TestCase, assert_array_almost_equal,
def test_iterative_refinements_dense(self):
    A = np.array([[1, 2, 3, 4, 0, 5, 0, 7], [0, 8, 7, 0, 1, 5, 9, 0], [1, 0, 0, 0, 0, 1, 2, 3]])
    test_points = ([1, 2, 3, 4, 5, 6, 7, 8], [1, 10, 3, 0, 1, 6, 7, 8], [1, 0, 0, 0, 0, 1, 2, 3 + 1e-10])
    for method in available_dense_methods:
        Z, LS, _ = projections(A, method, orth_tol=1e-18, max_refin=10)
        for z in test_points:
            x = Z.matvec(z)
            assert_allclose(A.dot(x), 0, rtol=0, atol=2.5e-14)
            assert_allclose(orthogonality(A, x), 0, rtol=0, atol=5e-16)