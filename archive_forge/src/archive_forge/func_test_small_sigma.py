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
@pytest.mark.parametrize('shape', ((20, 20), (20, 21), (21, 20)))
@pytest.mark.parametrize('dtype', (float, complex, np.float32))
def test_small_sigma(self, shape, dtype):
    if not has_propack:
        pytest.skip('PROPACK not enabled')
    if dtype == complex and self.solver == 'propack':
        pytest.skip('PROPACK unsupported for complex dtype')
    rng = np.random.default_rng(179847540)
    A = rng.random(shape).astype(dtype)
    u, _, vh = svd(A, full_matrices=False)
    if dtype == np.float32:
        e = 10.0
    else:
        e = 100.0
    t = e ** (-np.arange(len(vh))).astype(dtype)
    A = (u * t).dot(vh)
    k = 4
    u, s, vh = svds(A, k, solver=self.solver, maxiter=100)
    t = np.sum(s > 0)
    assert_equal(t, k)
    _check_svds_n(A, k, u, s, vh, atol=0.001, rtol=1.0, check_svd=False)