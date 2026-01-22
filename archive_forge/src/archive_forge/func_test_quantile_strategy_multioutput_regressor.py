import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_quantile_strategy_multioutput_regressor(global_random_seed):
    random_state = np.random.RandomState(seed=global_random_seed)
    X_learn = random_state.randn(10, 10)
    y_learn = random_state.randn(10, 5)
    median = np.median(y_learn, axis=0).reshape((1, -1))
    quantile_values = np.percentile(y_learn, axis=0, q=80).reshape((1, -1))
    X_test = random_state.randn(20, 10)
    y_test = random_state.randn(20, 5)
    est = DummyRegressor(strategy='quantile', quantile=0.5)
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)
    _check_equality_regressor(median, y_learn, y_pred_learn, y_test, y_pred_test)
    _check_behavior_2d(est)
    est = DummyRegressor(strategy='quantile', quantile=0.8)
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)
    _check_equality_regressor(quantile_values, y_learn, y_pred_learn, y_test, y_pred_test)
    _check_behavior_2d(est)