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
def test_quantile_transform_dense_toy():
    X = np.array([[0, 2, 2.6], [25, 4, 4.1], [50, 6, 2.3], [75, 8, 9.5], [100, 10, 0.1]])
    transformer = QuantileTransformer(n_quantiles=5)
    transformer.fit(X)
    X_trans = transformer.fit_transform(X)
    X_expected = np.tile(np.linspace(0, 1, num=5), (3, 1)).T
    assert_almost_equal(np.sort(X_trans, axis=0), X_expected)
    X_test = np.array([[-1, 1, 0], [101, 11, 10]])
    X_expected = np.array([[0, 0, 0], [1, 1, 1]])
    assert_array_almost_equal(transformer.transform(X_test), X_expected)
    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)