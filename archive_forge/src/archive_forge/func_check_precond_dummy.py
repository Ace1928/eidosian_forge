import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
import pytest
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from scipy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, SparseEfficiencyWarning, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def check_precond_dummy(solver, case):
    tol = 1e-08

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
    if solver is qmr:
        x, info = solver(A, b, M1=precond, M2=precond, x0=x0, tol=tol)
    else:
        x, info = solver(A, b, M=precond, x0=x0, tol=tol)
    assert info == 0
    assert_normclose(A @ x, b, tol)
    A = aslinearoperator(A)
    A.psolve = identity
    A.rpsolve = identity
    x, info = solver(A, b, x0=x0, tol=tol)
    assert info == 0
    assert_normclose(A @ x, b, tol=tol)