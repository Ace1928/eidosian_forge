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
@pytest.mark.parametrize('k', [-1, 0, 3, 4, 5, 1.5, '1'])
def test_svds_input_validation_k_1(self, k):
    rng = np.random.default_rng(0)
    A = rng.random((4, 3))
    if self.solver == 'propack' and k == 3:
        if not has_propack:
            pytest.skip('PROPACK not enabled')
        res = svds(A, k=k, solver=self.solver)
        _check_svds(A, k, *res, check_usvh_A=True, check_svd=True)
        return
    message = '`k` must be an integer satisfying'
    with pytest.raises(ValueError, match=message):
        svds(A, k=k, solver=self.solver)