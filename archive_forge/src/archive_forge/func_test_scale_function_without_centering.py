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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_scale_function_without_centering(csr_container):
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0
    X_csr = csr_container(X)
    X_scaled = scale(X, with_mean=False)
    assert not np.any(np.isnan(X_scaled))
    X_csr_scaled = scale(X_csr, with_mean=False)
    assert not np.any(np.isnan(X_csr_scaled.data))
    X_csc_scaled = scale(X_csr.tocsc(), with_mean=False)
    assert_array_almost_equal(X_scaled, X_csc_scaled.toarray())
    with pytest.raises(ValueError):
        scale(X_csr, with_mean=False, axis=1)
    assert_array_almost_equal(X_scaled.mean(axis=0), [0.0, -0.01, 2.24, -0.35, -0.78], 2)
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    assert X_scaled is not X
    X_csr_scaled_mean, X_csr_scaled_std = mean_variance_axis(X_csr_scaled, 0)
    assert_array_almost_equal(X_csr_scaled_mean, X_scaled.mean(axis=0))
    assert_array_almost_equal(X_csr_scaled_std, X_scaled.std(axis=0))
    X_csr_scaled = scale(X_csr, with_mean=False, with_std=False, copy=True)
    assert_array_almost_equal(X_csr.toarray(), X_csr_scaled.toarray())