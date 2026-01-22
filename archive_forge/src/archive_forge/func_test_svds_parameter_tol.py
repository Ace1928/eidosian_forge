import os
import re
import copy
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import pytest
from scipy.linalg import svd, null_space
from scipy.sparse import csc_matrix, issparse, spdiags, random
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg import svds
from scipy.sparse.linalg._eigen.arpack import ArpackNoConvergence
def test_svds_parameter_tol(self):
    if self.solver == 'propack':
        if not has_propack:
            pytest.skip('PROPACK not available')
    return
    n = 100
    k = 3
    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    A[A > 0.1] = 0
    A = A @ A.T
    _, s, _ = svd(A)
    A = csc_matrix(A)

    def err(tol):
        if self.solver == 'lobpcg' and tol == 0.0001:
            with pytest.warns(UserWarning, match='Exited at iteration'):
                _, s2, _ = svds(A, k=k, v0=np.ones(n), solver=self.solver, tol=tol)
        else:
            _, s2, _ = svds(A, k=k, v0=np.ones(n), solver=self.solver, tol=tol)
        return np.linalg.norm((s2 - s[k - 1::-1]) / s[k - 1::-1])
    tols = [0.0001, 0.01, 1.0]
    accuracies = {'propack': [1e-12, 1e-06, 0.0001], 'arpack': [2e-15, 1e-10, 1e-10], 'lobpcg': [1e-11, 0.001, 10]}
    for tol, accuracy in zip(tols, accuracies[self.solver]):
        error = err(tol)
        assert error < accuracy
        assert error > accuracy / 10