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
@pytest.mark.parametrize('method', ['box-cox', 'yeo-johnson'])
@pytest.mark.parametrize('standardize', [True, False])
def test_power_transformer_copy_False(method, standardize):
    X = X_1col
    if method == 'box-cox':
        X = np.abs(X)
    X_original = X.copy()
    assert X is not X_original
    assert_array_almost_equal(X, X_original)
    pt = PowerTransformer(method, standardize=standardize, copy=False)
    pt.fit(X)
    assert_array_almost_equal(X, X_original)
    X_trans = pt.transform(X)
    assert X_trans is X
    if method == 'box-cox':
        X = np.abs(X)
    X_trans = pt.fit_transform(X)
    assert X_trans is X
    X_inv_trans = pt.inverse_transform(X_trans)
    assert X_trans is X_inv_trans