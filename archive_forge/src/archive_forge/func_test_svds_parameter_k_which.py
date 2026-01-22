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
@pytest.mark.parametrize('k', [3, 5])
@pytest.mark.parametrize('which', ['LM', 'SM'])
def test_svds_parameter_k_which(self, k, which):
    if self.solver == 'propack':
        if not has_propack:
            pytest.skip('PROPACK not available')
    rng = np.random.default_rng(0)
    A = rng.random((10, 10))
    if self.solver == 'lobpcg':
        with pytest.warns(UserWarning, match='The problem size'):
            res = svds(A, k=k, which=which, solver=self.solver, random_state=0)
    else:
        res = svds(A, k=k, which=which, solver=self.solver, random_state=0)
    _check_svds(A, k, *res, which=which, atol=8e-10)