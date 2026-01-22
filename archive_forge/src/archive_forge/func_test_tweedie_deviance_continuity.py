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
def test_tweedie_deviance_continuity():
    n_samples = 100
    y_true = np.random.RandomState(0).rand(n_samples) + 0.1
    y_pred = np.random.RandomState(1).rand(n_samples) + 0.1
    assert_allclose(mean_tweedie_deviance(y_true, y_pred, power=0 - 1e-10), mean_tweedie_deviance(y_true, y_pred, power=0))
    assert_allclose(mean_tweedie_deviance(y_true, y_pred, power=1 + 1e-10), mean_tweedie_deviance(y_true, y_pred, power=1), atol=1e-06)
    assert_allclose(mean_tweedie_deviance(y_true, y_pred, power=2 - 1e-10), mean_tweedie_deviance(y_true, y_pred, power=2), atol=1e-06)
    assert_allclose(mean_tweedie_deviance(y_true, y_pred, power=2 + 1e-10), mean_tweedie_deviance(y_true, y_pred, power=2), atol=1e-06)