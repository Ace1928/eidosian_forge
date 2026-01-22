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
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_quantile_transform_iris(csc_container):
    X = iris.data
    transformer = QuantileTransformer(n_quantiles=30)
    X_trans = transformer.fit_transform(X)
    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)
    transformer = QuantileTransformer(n_quantiles=30, output_distribution='normal')
    X_trans = transformer.fit_transform(X)
    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)
    X_sparse = csc_container(X)
    X_sparse_tran = transformer.fit_transform(X_sparse)
    X_sparse_tran_inv = transformer.inverse_transform(X_sparse_tran)
    assert_array_almost_equal(X_sparse.toarray(), X_sparse_tran_inv.toarray())