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
def test_scaler_2d_arrays():
    rng = np.random.RandomState(0)
    n_features = 5
    n_samples = 4
    X = rng.randn(n_samples, n_features)
    X[:, 0] = 0.0
    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=True)
    assert not np.any(np.isnan(X_scaled))
    assert scaler.n_samples_seen_ == n_samples
    assert_array_almost_equal(X_scaled.mean(axis=0), n_features * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    assert X_scaled is not X
    X_scaled_back = scaler.inverse_transform(X_scaled)
    assert X_scaled_back is not X
    assert X_scaled_back is not X_scaled
    assert_array_almost_equal(X_scaled_back, X)
    X_scaled = scale(X, axis=1, with_std=False)
    assert not np.any(np.isnan(X_scaled))
    assert_array_almost_equal(X_scaled.mean(axis=1), n_samples * [0.0])
    X_scaled = scale(X, axis=1, with_std=True)
    assert not np.any(np.isnan(X_scaled))
    assert_array_almost_equal(X_scaled.mean(axis=1), n_samples * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=1), n_samples * [1.0])
    assert X_scaled is not X
    X_scaled = scaler.fit(X).transform(X, copy=False)
    assert not np.any(np.isnan(X_scaled))
    assert_array_almost_equal(X_scaled.mean(axis=0), n_features * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    assert X_scaled is X
    X = rng.randn(4, 5)
    X[:, 0] = 1.0
    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=True)
    assert not np.any(np.isnan(X_scaled))
    assert_array_almost_equal(X_scaled.mean(axis=0), n_features * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    assert X_scaled is not X