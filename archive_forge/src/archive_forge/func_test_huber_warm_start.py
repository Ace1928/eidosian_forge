import numpy as np
import pytest
from scipy import optimize
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, SGDRegressor
from sklearn.linear_model._huber import _huber_loss_and_gradient
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_huber_warm_start():
    X, y = make_regression_with_outliers()
    huber_warm = HuberRegressor(alpha=1.0, max_iter=10000, warm_start=True, tol=0.1)
    huber_warm.fit(X, y)
    huber_warm_coef = huber_warm.coef_.copy()
    huber_warm.fit(X, y)
    assert_array_almost_equal(huber_warm.coef_, huber_warm_coef, 1)
    assert huber_warm.n_iter_ == 0