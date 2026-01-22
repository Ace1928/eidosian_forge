import numpy as np
import pytest
from scipy import optimize
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, SGDRegressor
from sklearn.linear_model._huber import _huber_loss_and_gradient
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_huber_scaling_invariant():
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(fit_intercept=False, alpha=0.0)
    huber.fit(X, y)
    n_outliers_mask_1 = huber.outliers_
    assert not np.all(n_outliers_mask_1)
    huber.fit(X, 2.0 * y)
    n_outliers_mask_2 = huber.outliers_
    assert_array_equal(n_outliers_mask_2, n_outliers_mask_1)
    huber.fit(2.0 * X, 2.0 * y)
    n_outliers_mask_3 = huber.outliers_
    assert_array_equal(n_outliers_mask_3, n_outliers_mask_1)