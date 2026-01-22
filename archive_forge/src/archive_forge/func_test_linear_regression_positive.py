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
def test_linear_regression_positive():
    X = [[1], [2]]
    y = [1, 2]
    reg = LinearRegression(positive=True)
    reg.fit(X, y)
    assert_array_almost_equal(reg.coef_, [1])
    assert_array_almost_equal(reg.intercept_, [0])
    assert_array_almost_equal(reg.predict(X), [1, 2])
    X = [[1]]
    y = [0]
    reg = LinearRegression(positive=True)
    reg.fit(X, y)
    assert_allclose(reg.coef_, [0])
    assert_allclose(reg.intercept_, [0])
    assert_allclose(reg.predict(X), [0])