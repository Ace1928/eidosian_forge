import warnings
import numpy as np
import pytest
from scipy import linalg, sparse
from sklearn.datasets import load_iris, make_regression, make_sparse_uncorrelated
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import (
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
@pytest.mark.parametrize('to_copy', (True, False))
def test_preprocess_copy_data_no_checks(sparse_container, to_copy):
    X, y = make_regression()
    X[X < 2.5] = 0.0
    if sparse_container is not None:
        X = sparse_container(X)
    X_, y_, _, _, _ = _preprocess_data(X, y, fit_intercept=True, copy=to_copy, check_input=False)
    if to_copy and sparse_container is not None:
        assert not np.may_share_memory(X_.data, X.data)
    elif to_copy:
        assert not np.may_share_memory(X_, X)
    elif sparse_container is not None:
        assert np.may_share_memory(X_.data, X.data)
    else:
        assert np.may_share_memory(X_, X)