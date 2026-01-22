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
@pytest.mark.parametrize('n_samples', [10, 100, 10000])
@pytest.mark.parametrize('average', [1e-10, 1, 10000000000.0])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS + CSR_CONTAINERS)
def test_standard_scaler_near_constant_features(n_samples, sparse_container, average, dtype):
    scale_min, scale_max = (-30, 19)
    scales = np.array([10 ** i for i in range(scale_min, scale_max + 1)], dtype=dtype)
    n_features = scales.shape[0]
    X = np.empty((n_samples, n_features), dtype=dtype)
    X[:n_samples // 2, :] = average + scales
    X[n_samples // 2:, :] = average - scales
    X_array = X if sparse_container is None else sparse_container(X)
    scaler = StandardScaler(with_mean=False).fit(X_array)
    eps = np.finfo(np.float64).eps
    bounds = n_samples * eps * scales ** 2 + n_samples ** 2 * eps ** 2 * average ** 2
    within_bounds = scales ** 2 <= bounds
    assert np.any(within_bounds)
    assert all(scaler.var_[within_bounds] <= bounds[within_bounds])
    assert_allclose(scaler.scale_[within_bounds], 1.0)
    representable_diff = X[0, :] - X[-1, :] != 0
    assert_allclose(scaler.var_[np.logical_not(representable_diff)], 0)
    assert_allclose(scaler.scale_[np.logical_not(representable_diff)], 1)
    common_mask = np.logical_and(scales ** 2 > bounds, representable_diff)
    assert_allclose(scaler.scale_[common_mask], np.sqrt(scaler.var_)[common_mask])