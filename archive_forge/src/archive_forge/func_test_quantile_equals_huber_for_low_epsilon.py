import numpy as np
import pytest
from pytest import approx
from scipy.optimize import minimize
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.utils._testing import assert_allclose, skip_if_32bit
from sklearn.utils.fixes import (
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_quantile_equals_huber_for_low_epsilon(fit_intercept, default_solver):
    X, y = make_regression(n_samples=100, n_features=20, random_state=0, noise=1.0)
    alpha = 0.0001
    huber = HuberRegressor(epsilon=1 + 0.0001, alpha=alpha, fit_intercept=fit_intercept).fit(X, y)
    quant = QuantileRegressor(alpha=alpha, fit_intercept=fit_intercept, solver=default_solver).fit(X, y)
    assert_allclose(huber.coef_, quant.coef_, atol=0.1)
    if fit_intercept:
        assert huber.intercept_ == approx(quant.intercept_, abs=0.1)
        assert np.mean(y < quant.predict(X)) == approx(0.5, abs=0.1)