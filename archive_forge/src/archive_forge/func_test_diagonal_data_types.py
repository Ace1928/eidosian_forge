import itertools
import platform
import sys
import pytest
import numpy as np
from numpy import ones, r_, diag
from numpy.testing import (assert_almost_equal, assert_equal,
from scipy import sparse
from scipy.linalg import eig, eigh, toeplitz, orth
from scipy.sparse import spdiags, diags, eye, csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg
from scipy.sparse.linalg._eigen.lobpcg.lobpcg import _b_orthonormalize
from scipy._lib._util import np_long, np_ulong
@pytest.mark.slow
@pytest.mark.parametrize('n', [15])
@pytest.mark.parametrize('m', [1, 2])
@pytest.mark.filterwarnings('ignore:Exited at iteration')
@pytest.mark.filterwarnings('ignore:Exited postprocessing')
def test_diagonal_data_types(n, m):
    """Check lobpcg for diagonal matrices for all matrix types.
    Constraints are imposed, so a dense eigensolver eig cannot run.
    """
    rnd = np.random.RandomState(0)
    vals = np.arange(1, n + 1)
    list_sparse_format = ['coo']
    sparse_formats = len(list_sparse_format)
    for s_f_i, s_f in enumerate(list_sparse_format):
        As64 = diags([vals * vals], [0], (n, n), format=s_f)
        As32 = As64.astype(np.float32)
        Af64 = As64.toarray()
        Af32 = Af64.astype(np.float32)

        def As32f(x):
            return As32 @ x
        As32LO = LinearOperator(matvec=As32f, matmat=As32f, shape=(n, n), dtype=As32.dtype)
        listA = [Af64, As64, Af32, As32, As32f, As32LO, lambda v: As32 @ v]
        Bs64 = diags([vals], [0], (n, n), format=s_f)
        Bf64 = Bs64.toarray()
        Bs32 = Bs64.astype(np.float32)

        def Bs32f(x):
            return Bs32 @ x
        Bs32LO = LinearOperator(matvec=Bs32f, matmat=Bs32f, shape=(n, n), dtype=Bs32.dtype)
        listB = [Bf64, Bs64, Bs32, Bs32f, Bs32LO, lambda v: Bs32 @ v]
        Ms64 = diags([1.0 / vals], [0], (n, n), format=s_f)

        def Ms64precond(x):
            return Ms64 @ x
        Ms64precondLO = LinearOperator(matvec=Ms64precond, matmat=Ms64precond, shape=(n, n), dtype=Ms64.dtype)
        Mf64 = Ms64.toarray()

        def Mf64precond(x):
            return Mf64 @ x
        Mf64precondLO = LinearOperator(matvec=Mf64precond, matmat=Mf64precond, shape=(n, n), dtype=Mf64.dtype)
        Ms32 = Ms64.astype(np.float32)

        def Ms32precond(x):
            return Ms32 @ x
        Ms32precondLO = LinearOperator(matvec=Ms32precond, matmat=Ms32precond, shape=(n, n), dtype=Ms32.dtype)
        Mf32 = Ms32.toarray()

        def Mf32precond(x):
            return Mf32 @ x
        Mf32precondLO = LinearOperator(matvec=Mf32precond, matmat=Mf32precond, shape=(n, n), dtype=Mf32.dtype)
        listM = [None, Ms64, Ms64precondLO, Mf64precondLO, Ms64precond, Ms32, Ms32precondLO, Mf32precondLO, Ms32precond]
        Xf64 = rnd.random((n, m))
        Xf32 = Xf64.astype(np.float32)
        listX = [Xf64, Xf32]
        m_excluded = 3
        Yf64 = np.eye(n, m_excluded, dtype=float)
        Yf32 = np.eye(n, m_excluded, dtype=np.float32)
        listY = [Yf64, Yf32]
        tests = list(itertools.product(listA, listB, listM, listX, listY))
        if s_f_i > 0:
            tests = tests[s_f_i - 1::sparse_formats - 1]
        for A, B, M, X, Y in tests:
            eigvals, _ = lobpcg(A, X, B=B, M=M, Y=Y, tol=0.0001, maxiter=100, largest=False)
            assert_allclose(eigvals, np.arange(1 + m_excluded, 1 + m_excluded + m), atol=1e-05)