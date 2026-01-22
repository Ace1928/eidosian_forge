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
def test_svd_linop(self):
    solver = self.solver
    if self.solver == 'propack':
        if not has_propack:
            pytest.skip('PROPACK not available')
    nmks = [(6, 7, 3), (9, 5, 4), (10, 8, 5)]

    def reorder(args):
        U, s, VH = args
        j = np.argsort(s)
        return (U[:, j], s[j], VH[j, :])
    for n, m, k in nmks:
        A = np.random.RandomState(52).randn(n, m)
        L = CheckingLinearOperator(A)
        if solver == 'propack':
            v0 = np.ones(n)
        else:
            v0 = np.ones(min(A.shape))
        if solver == 'lobpcg':
            with pytest.warns(UserWarning, match='The problem size'):
                U1, s1, VH1 = reorder(svds(A, k, v0=v0, solver=solver))
                U2, s2, VH2 = reorder(svds(L, k, v0=v0, solver=solver))
        else:
            U1, s1, VH1 = reorder(svds(A, k, v0=v0, solver=solver))
            U2, s2, VH2 = reorder(svds(L, k, v0=v0, solver=solver))
        assert_allclose(np.abs(U1), np.abs(U2))
        assert_allclose(s1, s2)
        assert_allclose(np.abs(VH1), np.abs(VH2))
        assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)), np.dot(U2, np.dot(np.diag(s2), VH2)))
        A = np.random.RandomState(1909).randn(n, m)
        L = CheckingLinearOperator(A)
        kwargs = {'v0': v0} if solver not in {None, 'arpack'} else {}
        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match='The problem size'):
                U1, s1, VH1 = reorder(svds(A, k, which='SM', solver=solver, **kwargs))
                U2, s2, VH2 = reorder(svds(L, k, which='SM', solver=solver, **kwargs))
        else:
            U1, s1, VH1 = reorder(svds(A, k, which='SM', solver=solver, **kwargs))
            U2, s2, VH2 = reorder(svds(L, k, which='SM', solver=solver, **kwargs))
        assert_allclose(np.abs(U1), np.abs(U2))
        assert_allclose(s1 + 1, s2 + 1)
        assert_allclose(np.abs(VH1), np.abs(VH2))
        assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)), np.dot(U2, np.dot(np.diag(s2), VH2)))
        if k < min(n, m) - 1:
            for dt, eps in [(complex, 1e-07), (np.complex64, 0.001)]:
                if self.solver == 'propack' and np.intp(0).itemsize < 8:
                    pytest.skip('PROPACK complex-valued SVD methods not available for 32-bit builds')
                rng = np.random.RandomState(1648)
                A = (rng.randn(n, m) + 1j * rng.randn(n, m)).astype(dt)
                L = CheckingLinearOperator(A)
                if self.solver == 'lobpcg':
                    with pytest.warns(UserWarning, match='The problem size'):
                        U1, s1, VH1 = reorder(svds(A, k, which='LM', solver=solver))
                        U2, s2, VH2 = reorder(svds(L, k, which='LM', solver=solver))
                else:
                    U1, s1, VH1 = reorder(svds(A, k, which='LM', solver=solver))
                    U2, s2, VH2 = reorder(svds(L, k, which='LM', solver=solver))
                assert_allclose(np.abs(U1), np.abs(U2), rtol=eps)
                assert_allclose(s1, s2, rtol=eps)
                assert_allclose(np.abs(VH1), np.abs(VH2), rtol=eps)
                assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)), np.dot(U2, np.dot(np.diag(s2), VH2)), rtol=eps)