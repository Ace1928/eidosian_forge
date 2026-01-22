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
def test_svd_random_state(self):
    if self.solver == 'propack':
        if not has_propack:
            pytest.skip('PROPACK not available')
    n = 100
    k = 1
    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    res1a = svds(A, k, solver=self.solver, random_state=0)
    res2a = svds(A, k, solver=self.solver, random_state=0)
    for idx in range(3):
        assert_allclose(res1a[idx], res2a[idx], rtol=1e-15, atol=2e-16)
    _check_svds(A, k, *res1a)
    res1b = svds(A, k, solver=self.solver, random_state=1)
    res2b = svds(A, k, solver=self.solver, random_state=1)
    for idx in range(3):
        assert_allclose(res1b[idx], res2b[idx], rtol=1e-15, atol=2e-16)
    _check_svds(A, k, *res1b)
    message = 'Arrays are not equal'
    with pytest.raises(AssertionError, match=message):
        assert_equal(res1a, res1b)