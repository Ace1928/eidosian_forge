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
def test_raises_value_error_if_positive_and_sparse():
    error_msg = 'Sparse data was passed for X, but dense data is required.'
    X = sparse.eye(10)
    y = np.ones(10)
    reg = LinearRegression(positive=True)
    with pytest.raises(TypeError, match=error_msg):
        reg.fit(X, y)