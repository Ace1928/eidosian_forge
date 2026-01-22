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
def test_precond_dummy(case):
    if not case.convergence:
        pytest.skip('Solver - Breakdown case, see gh-8829')
    rtol = 1e-08

    def identity(b, which=None):
        """trivial preconditioner"""
        return b
    A = case.A
    M, N = A.shape
    diagOfA = A.diagonal()
    if np.count_nonzero(diagOfA) == len(diagOfA):
        spdiags([1.0 / diagOfA], [0], M, N)
    b = case.b
    x0 = 0 * b
    precond = LinearOperator(A.shape, identity, rmatvec=identity)
    if case.solver is qmr:
        x, info = case.solver(A, b, M1=precond, M2=precond, x0=x0, rtol=rtol)
    else:
        x, info = case.solver(A, b, M=precond, x0=x0, rtol=rtol)
    assert info == 0
    assert norm(A @ x - b) <= norm(b) * rtol
    A = aslinearoperator(A)
    A.psolve = identity
    A.rpsolve = identity
    x, info = case.solver(A, b, x0=x0, rtol=rtol)
    assert info == 0
    assert norm(A @ x - b) <= norm(b) * rtol