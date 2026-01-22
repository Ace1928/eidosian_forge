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
def test_power_transformer_2d():
    X = np.abs(X_2d)
    for standardize in [True, False]:
        pt = PowerTransformer(method='box-cox', standardize=standardize)
        X_trans_class = pt.fit_transform(X)
        X_trans_func = power_transform(X, method='box-cox', standardize=standardize)
        for X_trans in [X_trans_class, X_trans_func]:
            for j in range(X_trans.shape[1]):
                X_expected, lmbda = stats.boxcox(X[:, j].flatten())
                if standardize:
                    X_expected = scale(X_expected)
                assert_almost_equal(X_trans[:, j], X_expected)
                assert_almost_equal(lmbda, pt.lambdas_[j])
            X_inv = pt.inverse_transform(X_trans)
            assert_array_almost_equal(X_inv, X)
        assert len(pt.lambdas_) == X.shape[1]
        assert isinstance(pt.lambdas_, np.ndarray)