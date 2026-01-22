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
def test_zero_rhs(solver):
    rng = np.random.default_rng(1684414984100503)
    A = rng.random(size=[10, 10])
    A = A @ A.T + 10 * np.eye(10)
    b = np.zeros(10)
    tols = np.r_[np.logspace(-10, 2, 7)]
    for tol in tols:
        x, info = solver(A, b, rtol=tol)
        assert info == 0
        assert_allclose(x, 0.0, atol=1e-15)
        x, info = solver(A, b, rtol=tol, x0=ones(10))
        assert info == 0
        assert_allclose(x, 0.0, atol=tol)
        if solver is not minres:
            x, info = solver(A, b, rtol=tol, atol=0, x0=ones(10))
            if info == 0:
                assert_allclose(x, 0)
            x, info = solver(A, b, rtol=tol, atol=tol)
            assert info == 0
            assert_allclose(x, 0, atol=1e-300)
            x, info = solver(A, b, rtol=tol, atol=0)
            assert info == 0
            assert_allclose(x, 0, atol=1e-300)