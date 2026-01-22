import numpy as np
import pytest
import scipy.sparse as sp
from numpy.random import RandomState
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import linalg
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.sparsefuncs import (
from sklearn.utils.sparsefuncs_fast import (
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('sparse_constructor', CSC_CONTAINERS + CSR_CONTAINERS)
def test_incr_mean_variance_axis_ignore_nan(axis, sparse_constructor):
    old_means = np.array([535.0, 535.0, 535.0, 535.0])
    old_variances = np.array([4225.0, 4225.0, 4225.0, 4225.0])
    old_sample_count = np.array([2, 2, 2, 2], dtype=np.int64)
    X = sparse_constructor(np.array([[170, 170, 170, 170], [430, 430, 430, 430], [300, 300, 300, 300]]))
    X_nan = sparse_constructor(np.array([[170, np.nan, 170, 170], [np.nan, 170, 430, 430], [430, 430, np.nan, 300], [300, 300, 300, np.nan]]))
    if axis:
        X = X.T
        X_nan = X_nan.T
    X_means, X_vars, X_sample_count = incr_mean_variance_axis(X, axis=axis, last_mean=old_means.copy(), last_var=old_variances.copy(), last_n=old_sample_count.copy())
    X_nan_means, X_nan_vars, X_nan_sample_count = incr_mean_variance_axis(X_nan, axis=axis, last_mean=old_means.copy(), last_var=old_variances.copy(), last_n=old_sample_count.copy())
    assert_allclose(X_nan_means, X_means)
    assert_allclose(X_nan_vars, X_vars)
    assert_allclose(X_nan_sample_count, X_sample_count)