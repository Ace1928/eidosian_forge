from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import optimize
from scipy.special import factorial, xlogy
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._regression import _check_reg_targets
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import (
def test_dummy_quantile_parameter_tuning():
    n_samples = 1000
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, 5))
    y = rng.exponential(size=n_samples)
    all_quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for alpha in all_quantiles:
        neg_mean_pinball_loss = make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False)
        regressor = DummyRegressor(strategy='quantile', quantile=0.25)
        grid_search = GridSearchCV(regressor, param_grid=dict(quantile=all_quantiles), scoring=neg_mean_pinball_loss).fit(X, y)
        assert grid_search.best_params_['quantile'] == pytest.approx(alpha)