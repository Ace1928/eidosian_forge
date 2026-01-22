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
def test_regression_multioutput_array():
    y_true = [[1, 2], [2.5, -1], [4.5, 3], [5, 7]]
    y_pred = [[1, 1], [2, -1], [5, 4], [5, 6.5]]
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    pbl = mean_pinball_loss(y_true, y_pred, multioutput='raw_values')
    mape = mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
    r = r2_score(y_true, y_pred, multioutput='raw_values')
    evs = explained_variance_score(y_true, y_pred, multioutput='raw_values')
    d2ps = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput='raw_values')
    evs2 = explained_variance_score(y_true, y_pred, multioutput='raw_values', force_finite=False)
    assert_array_almost_equal(mse, [0.125, 0.5625], decimal=2)
    assert_array_almost_equal(mae, [0.25, 0.625], decimal=2)
    assert_array_almost_equal(pbl, [0.25 / 2, 0.625 / 2], decimal=2)
    assert_array_almost_equal(mape, [0.0778, 0.2262], decimal=2)
    assert_array_almost_equal(r, [0.95, 0.93], decimal=2)
    assert_array_almost_equal(evs, [0.95, 0.93], decimal=2)
    assert_array_almost_equal(d2ps, [0.833, 0.722], decimal=2)
    assert_array_almost_equal(evs2, [0.95, 0.93], decimal=2)
    y_true = [[0, 0]] * 4
    y_pred = [[1, 1]] * 4
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    pbl = mean_pinball_loss(y_true, y_pred, multioutput='raw_values')
    r = r2_score(y_true, y_pred, multioutput='raw_values')
    d2ps = d2_pinball_score(y_true, y_pred, multioutput='raw_values')
    assert_array_almost_equal(mse, [1.0, 1.0], decimal=2)
    assert_array_almost_equal(mae, [1.0, 1.0], decimal=2)
    assert_array_almost_equal(pbl, [0.5, 0.5], decimal=2)
    assert_array_almost_equal(r, [0.0, 0.0], decimal=2)
    assert_array_almost_equal(d2ps, [0.0, 0.0], decimal=2)
    r = r2_score([[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput='raw_values')
    assert_array_almost_equal(r, [0, -3.5], decimal=2)
    assert np.mean(r) == r2_score([[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput='uniform_average')
    evs = explained_variance_score([[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput='raw_values')
    assert_array_almost_equal(evs, [0, -1.25], decimal=2)
    evs2 = explained_variance_score([[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput='raw_values', force_finite=False)
    assert_array_almost_equal(evs2, [-np.inf, -1.25], decimal=2)
    y_true = [[1, 3], [1, 2]]
    y_pred = [[1, 4], [1, 1]]
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    assert_array_almost_equal(r2, [1.0, -3.0], decimal=2)
    assert np.mean(r2) == r2_score(y_true, y_pred, multioutput='uniform_average')
    r22 = r2_score(y_true, y_pred, multioutput='raw_values', force_finite=False)
    assert_array_almost_equal(r22, [np.nan, -3.0], decimal=2)
    assert_almost_equal(np.mean(r22), r2_score(y_true, y_pred, multioutput='uniform_average', force_finite=False))
    evs = explained_variance_score(y_true, y_pred, multioutput='raw_values')
    assert_array_almost_equal(evs, [1.0, -3.0], decimal=2)
    assert np.mean(evs) == explained_variance_score(y_true, y_pred)
    d2ps = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput='raw_values')
    assert_array_almost_equal(d2ps, [1.0, -1.0], decimal=2)
    evs2 = explained_variance_score(y_true, y_pred, multioutput='raw_values', force_finite=False)
    assert_array_almost_equal(evs2, [np.nan, -3.0], decimal=2)
    assert_almost_equal(np.mean(evs2), explained_variance_score(y_true, y_pred, force_finite=False))
    y_true = np.array([[0.5, 1], [1, 2], [7, 6]])
    y_pred = np.array([[0.5, 2], [1, 2.5], [8, 8]])
    msle = mean_squared_log_error(y_true, y_pred, multioutput='raw_values')
    msle2 = mean_squared_error(np.log(1 + y_true), np.log(1 + y_pred), multioutput='raw_values')
    assert_array_almost_equal(msle, msle2, decimal=2)