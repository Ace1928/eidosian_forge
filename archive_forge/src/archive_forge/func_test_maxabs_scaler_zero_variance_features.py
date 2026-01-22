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
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS)
def test_maxabs_scaler_zero_variance_features(sparse_container):
    X = [[0.0, 1.0, +0.5], [0.0, 1.0, -0.3], [0.0, 1.0, +1.5], [0.0, 0.0, +0.0]]
    scaler = MaxAbsScaler()
    X_trans = scaler.fit_transform(X)
    X_expected = [[0.0, 1.0, 1.0 / 3.0], [0.0, 1.0, -0.2], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    assert_array_almost_equal(X_trans, X_expected)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)
    X_new = [[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]]
    X_trans_new = scaler.transform(X_new)
    X_expected_new = [[+0.0, 2.0, 1.0 / 3.0], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.0]]
    assert_array_almost_equal(X_trans_new, X_expected_new, decimal=2)
    X_trans = maxabs_scale(X)
    assert_array_almost_equal(X_trans, X_expected)
    X_sparse = sparse_container(X)
    X_trans_sparse = scaler.fit_transform(X_sparse)
    X_expected = [[0.0, 1.0, 1.0 / 3.0], [0.0, 1.0, -0.2], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    assert_array_almost_equal(X_trans_sparse.toarray(), X_expected)
    X_trans_sparse_inv = scaler.inverse_transform(X_trans_sparse)
    assert_array_almost_equal(X, X_trans_sparse_inv.toarray())