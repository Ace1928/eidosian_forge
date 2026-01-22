import numpy as np
import pytest
from scipy import optimize
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, SGDRegressor
from sklearn.linear_model._huber import _huber_loss_and_gradient
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_huber_sample_weights(csr_container):
    X, y = make_regression_with_outliers()
    huber = HuberRegressor()
    huber.fit(X, y)
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_
    scale = max(np.mean(np.abs(huber.coef_)), np.mean(np.abs(huber.intercept_)))
    huber.fit(X, y, sample_weight=np.ones(y.shape[0]))
    assert_array_almost_equal(huber.coef_ / scale, huber_coef / scale)
    assert_array_almost_equal(huber.intercept_ / scale, huber_intercept / scale)
    X, y = make_regression_with_outliers(n_samples=5, n_features=20)
    X_new = np.vstack((X, np.vstack((X[1], X[1], X[3]))))
    y_new = np.concatenate((y, [y[1]], [y[1]], [y[3]]))
    huber.fit(X_new, y_new)
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_
    sample_weight = np.ones(X.shape[0])
    sample_weight[1] = 3
    sample_weight[3] = 2
    huber.fit(X, y, sample_weight=sample_weight)
    assert_array_almost_equal(huber.coef_ / scale, huber_coef / scale)
    assert_array_almost_equal(huber.intercept_ / scale, huber_intercept / scale)
    X_csr = csr_container(X)
    huber_sparse = HuberRegressor()
    huber_sparse.fit(X_csr, y, sample_weight=sample_weight)
    assert_array_almost_equal(huber_sparse.coef_ / scale, huber_coef / scale)