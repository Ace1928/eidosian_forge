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
@pytest.mark.parametrize('rsv', (True, False, 'u', 'vh'))
@pytest.mark.parametrize('shape', ((5, 7), (6, 6), (7, 5)))
def test_svd_return_singular_vectors(self, rsv, shape):
    if self.solver == 'propack':
        if not has_propack:
            pytest.skip('PROPACK not available')
    rng = np.random.default_rng(0)
    A = rng.random(shape)
    k = 2
    M, N = shape
    u, s, vh = sorted_svd(A, k)
    respect_u = True if self.solver == 'propack' else M <= N
    respect_vh = True if self.solver == 'propack' else M > N
    if self.solver == 'lobpcg':
        with pytest.warns(UserWarning, match='The problem size'):
            if rsv is False:
                s2 = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
                assert_allclose(s2, s)
            elif rsv == 'u' and respect_u:
                u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
                assert_allclose(np.abs(u2), np.abs(u))
                assert_allclose(s2, s)
                assert vh2 is None
            elif rsv == 'vh' and respect_vh:
                u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
                assert u2 is None
                assert_allclose(s2, s)
                assert_allclose(np.abs(vh2), np.abs(vh))
            else:
                u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
                if u2 is not None:
                    assert_allclose(np.abs(u2), np.abs(u))
                assert_allclose(s2, s)
                if vh2 is not None:
                    assert_allclose(np.abs(vh2), np.abs(vh))
    elif rsv is False:
        s2 = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
        assert_allclose(s2, s)
    elif rsv == 'u' and respect_u:
        u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
        assert_allclose(np.abs(u2), np.abs(u))
        assert_allclose(s2, s)
        assert vh2 is None
    elif rsv == 'vh' and respect_vh:
        u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
        assert u2 is None
        assert_allclose(s2, s)
        assert_allclose(np.abs(vh2), np.abs(vh))
    else:
        u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
        if u2 is not None:
            assert_allclose(np.abs(u2), np.abs(u))
        assert_allclose(s2, s)
        if vh2 is not None:
            assert_allclose(np.abs(vh2), np.abs(vh))