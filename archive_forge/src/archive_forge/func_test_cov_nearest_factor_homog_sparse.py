import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
@pytest.mark.parametrize('dm', [1, 2])
def test_cov_nearest_factor_homog_sparse(self, dm):
    d = 100
    X = np.zeros((d, dm), dtype=np.float64)
    x = np.linspace(0, 2 * np.pi, d)
    for j in range(dm):
        X[:, j] = np.sin(x * (j + 1))
    mat = np.dot(X, X.T)
    np.fill_diagonal(mat, np.diag(mat) + 3.1)
    rslt = cov_nearest_factor_homog(mat, dm)
    mat1 = rslt.to_matrix()
    smat = sparse.csr_matrix(mat)
    rslt = cov_nearest_factor_homog(smat, dm)
    mat2 = rslt.to_matrix()
    assert_allclose(mat1, mat2, rtol=0.25, atol=0.001)