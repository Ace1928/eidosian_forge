import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
def test_standard_scaler_1d():
    for X in [X_1row, X_1col, X_list_1row, X_list_1row]:
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=True)
        if isinstance(X, list):
            X = np.array(X)
        if _check_dim_1axis(X) == 1:
            assert_almost_equal(scaler.mean_, X.ravel())
            assert_almost_equal(scaler.scale_, np.ones(n_features))
            assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros_like(n_features))
            assert_array_almost_equal(X_scaled.std(axis=0), np.zeros_like(n_features))
        else:
            assert_almost_equal(scaler.mean_, X.mean())
            assert_almost_equal(scaler.scale_, X.std())
            assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros_like(n_features))
            assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
            assert_array_almost_equal(X_scaled.std(axis=0), 1.0)
        assert scaler.n_samples_seen_ == X.shape[0]
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert_array_almost_equal(X_scaled_back, X)
    X = np.ones((5, 1))
    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=True)
    assert_almost_equal(scaler.mean_, 1.0)
    assert_almost_equal(scaler.scale_, 1.0)
    assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
    assert_array_almost_equal(X_scaled.std(axis=0), 0.0)
    assert scaler.n_samples_seen_ == X.shape[0]