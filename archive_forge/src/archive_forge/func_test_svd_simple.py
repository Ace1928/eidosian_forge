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
@pytest.mark.filterwarnings('ignore:k >= N - 1', reason='needed to demonstrate #16725')
@pytest.mark.parametrize('A', (A1, A2))
@pytest.mark.parametrize('k', range(1, 5))
@pytest.mark.parametrize('real', (True, False))
@pytest.mark.parametrize('transpose', (False, True))
@pytest.mark.parametrize('lo_type', (np.asarray, csc_matrix, aslinearoperator))
def test_svd_simple(self, A, k, real, transpose, lo_type):
    if self.solver == 'propack':
        if not has_propack:
            pytest.skip('PROPACK not available')
    A = np.asarray(A)
    A = np.real(A) if real else A
    A = A.T if transpose else A
    A2 = lo_type(A)
    if k > min(A.shape):
        pytest.skip('`k` cannot be greater than `min(A.shape)`')
    if self.solver != 'propack' and k >= min(A.shape):
        pytest.skip('Only PROPACK supports complete SVD')
    if self.solver == 'arpack' and (not real) and (k == min(A.shape) - 1):
        pytest.skip('#16725')
    if self.solver == 'propack' and (np.intp(0).itemsize < 8 and (not real)):
        pytest.skip('PROPACK complex-valued SVD methods not available for 32-bit builds')
    if self.solver == 'lobpcg':
        with pytest.warns(UserWarning, match='The problem size'):
            u, s, vh = svds(A2, k, solver=self.solver)
    else:
        u, s, vh = svds(A2, k, solver=self.solver)
    _check_svds(A, k, u, s, vh, atol=3e-10)