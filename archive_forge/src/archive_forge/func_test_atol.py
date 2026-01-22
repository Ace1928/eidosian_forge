import itertools
import platform
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def test_atol(solver):
    if solver in (minres, tfqmr):
        pytest.skip('TODO: Add atol to minres/tfqmr')
    rng = np.random.default_rng(168441431005389)
    A = rng.uniform(size=[10, 10])
    A = A @ A.T + 10 * np.eye(10)
    b = 1000.0 * rng.uniform(size=10)
    b_norm = np.linalg.norm(b)
    tols = np.r_[0, np.logspace(-9, 2, 7), np.inf]
    M0 = rng.standard_normal(size=(10, 10))
    M0 = M0 @ M0.T
    Ms = [None, 1e-06 * M0, 1000000.0 * M0]
    for M, rtol, atol in itertools.product(Ms, tols, tols):
        if rtol == 0 and atol == 0:
            continue
        if solver is qmr:
            if M is not None:
                M = aslinearoperator(M)
                M2 = aslinearoperator(np.eye(10))
            else:
                M2 = None
            x, info = solver(A, b, M1=M, M2=M2, rtol=rtol, atol=atol)
        else:
            x, info = solver(A, b, M=M, rtol=rtol, atol=atol)
        assert info == 0
        residual = A @ x - b
        err = np.linalg.norm(residual)
        atol2 = rtol * b_norm
        assert err <= 1.00025 * max(atol, atol2)