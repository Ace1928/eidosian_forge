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
def test_dtype_preprocess_data(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 200
    n_features = 2
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    X_32 = np.asarray(X, dtype=np.float32)
    y_32 = np.asarray(y, dtype=np.float32)
    X_64 = np.asarray(X, dtype=np.float64)
    y_64 = np.asarray(y, dtype=np.float64)
    for fit_intercept in [True, False]:
        Xt_32, yt_32, X_mean_32, y_mean_32, X_scale_32 = _preprocess_data(X_32, y_32, fit_intercept=fit_intercept)
        Xt_64, yt_64, X_mean_64, y_mean_64, X_scale_64 = _preprocess_data(X_64, y_64, fit_intercept=fit_intercept)
        Xt_3264, yt_3264, X_mean_3264, y_mean_3264, X_scale_3264 = _preprocess_data(X_32, y_64, fit_intercept=fit_intercept)
        Xt_6432, yt_6432, X_mean_6432, y_mean_6432, X_scale_6432 = _preprocess_data(X_64, y_32, fit_intercept=fit_intercept)
        assert Xt_32.dtype == np.float32
        assert yt_32.dtype == np.float32
        assert X_mean_32.dtype == np.float32
        assert y_mean_32.dtype == np.float32
        assert X_scale_32.dtype == np.float32
        assert Xt_64.dtype == np.float64
        assert yt_64.dtype == np.float64
        assert X_mean_64.dtype == np.float64
        assert y_mean_64.dtype == np.float64
        assert X_scale_64.dtype == np.float64
        assert Xt_3264.dtype == np.float32
        assert yt_3264.dtype == np.float32
        assert X_mean_3264.dtype == np.float32
        assert y_mean_3264.dtype == np.float32
        assert X_scale_3264.dtype == np.float32
        assert Xt_6432.dtype == np.float64
        assert yt_6432.dtype == np.float64
        assert X_mean_6432.dtype == np.float64
        assert y_mean_6432.dtype == np.float64
        assert X_scale_6432.dtype == np.float64
        assert X_32.dtype == np.float32
        assert y_32.dtype == np.float32
        assert X_64.dtype == np.float64
        assert y_64.dtype == np.float64
        assert_array_almost_equal(Xt_32, Xt_64)
        assert_array_almost_equal(yt_32, yt_64)
        assert_array_almost_equal(X_mean_32, X_mean_64)
        assert_array_almost_equal(y_mean_32, y_mean_64)
        assert_array_almost_equal(X_scale_32, X_scale_64)