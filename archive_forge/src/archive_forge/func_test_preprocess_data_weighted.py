import warnings
import numpy as np
import pytest
from scipy import linalg, sparse
from sklearn.datasets import load_iris, make_regression, make_sparse_uncorrelated
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import (
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
def test_preprocess_data_weighted(sparse_container, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 200
    n_features = 4
    X = rng.rand(n_samples, n_features)
    X[X < 0.5] = 0.0
    X[:, 0] *= 10
    X[:, 2] = 1.0
    X[:, 3] = 0.0
    y = rng.rand(n_samples)
    sample_weight = rng.rand(n_samples)
    expected_X_mean = np.average(X, axis=0, weights=sample_weight)
    expected_y_mean = np.average(y, axis=0, weights=sample_weight)
    X_sample_weight_avg = np.average(X, weights=sample_weight, axis=0)
    X_sample_weight_var = np.average((X - X_sample_weight_avg) ** 2, weights=sample_weight, axis=0)
    constant_mask = X_sample_weight_var < 10 * np.finfo(X.dtype).eps
    assert_array_equal(constant_mask, [0, 0, 1, 1])
    expected_X_scale = np.sqrt(X_sample_weight_var) * np.sqrt(sample_weight.sum())
    expected_X_scale[constant_mask] = 1
    if sparse_container is not None:
        X = sparse_container(X)
    Xt, yt, X_mean, y_mean, X_scale = _preprocess_data(X, y, fit_intercept=True, sample_weight=sample_weight)
    assert_array_almost_equal(X_mean, expected_X_mean)
    assert_array_almost_equal(y_mean, expected_y_mean)
    assert_array_almost_equal(X_scale, np.ones(n_features))
    if sparse_container is not None:
        assert_array_almost_equal(Xt.toarray(), X.toarray())
    else:
        assert_array_almost_equal(Xt, X - expected_X_mean)
    assert_array_almost_equal(yt, y - expected_y_mean)