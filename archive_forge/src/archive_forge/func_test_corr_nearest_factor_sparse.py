import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
@pytest.mark.slow
@pytest.mark.parametrize('dm', [1, 2])
def test_corr_nearest_factor_sparse(self, dm):
    d = 200
    X = np.zeros((d, dm), dtype=np.float64)
    x = np.linspace(0, 2 * np.pi, d)
    rs = np.random.RandomState(10)
    for j in range(dm):
        X[:, j] = np.sin(x * (j + 1)) + rs.randn(d)
    _project_correlation_factors(X)
    X *= 0.7
    mat = np.dot(X, X.T)
    np.fill_diagonal(mat, 1)
    mat.flat[np.abs(mat.flat) < 0.35] = 0.0
    smat = sparse.csr_matrix(mat)
    dense_rslt = corr_nearest_factor(mat, dm, maxiter=10000)
    sparse_rslt = corr_nearest_factor(smat, dm, maxiter=10000)
    mat_dense = dense_rslt.corr.to_matrix()
    mat_sparse = sparse_rslt.corr.to_matrix()
    assert dense_rslt.Converged is sparse_rslt.Converged
    assert dense_rslt.Converged is True
    assert_allclose(mat_dense, mat_sparse, rtol=0.25, atol=0.001)