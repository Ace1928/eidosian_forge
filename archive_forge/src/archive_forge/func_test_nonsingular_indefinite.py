import numpy as np # noqa F403
import scipy.sparse as spar
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities import linalg as lau
def test_nonsingular_indefinite(self):
    np.random.seed(0)
    n = 5
    diag = np.random.rand(n) + 0.1
    diag[n - 1] = -1
    offdiag = np.min(np.abs(diag)) * np.ones(n - 1) / 2
    A = spar.diags([offdiag, diag, offdiag], [-1, 0, 1])
    with self.assertRaises(ValueError, msg=lau.SparseCholeskyMessages.INDEFINITE):
        lau.sparse_cholesky(A, 0.0)