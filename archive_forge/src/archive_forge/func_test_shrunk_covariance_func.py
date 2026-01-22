import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import (
from sklearn.covariance._shrunk_covariance import _ledoit_wolf
from sklearn.utils._testing import (
from .._shrunk_covariance import _oas
@pytest.mark.parametrize('n_matrices', [1, 3])
def test_shrunk_covariance_func(n_matrices):
    """Check `shrunk_covariance` function."""
    n_features = 2
    cov = np.ones((n_features, n_features))
    cov_target = np.array([[1, 0.5], [0.5, 1]])
    if n_matrices > 1:
        cov = np.repeat(cov[np.newaxis, ...], n_matrices, axis=0)
        cov_target = np.repeat(cov_target[np.newaxis, ...], n_matrices, axis=0)
    cov_shrunk = shrunk_covariance(cov, 0.5)
    assert_allclose(cov_shrunk, cov_target)