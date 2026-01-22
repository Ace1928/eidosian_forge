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
def test_root_mean_squared_error_multioutput_raw_value():
    mse = mean_squared_error([[1]], [[10]], multioutput='raw_values')
    rmse = root_mean_squared_error([[1]], [[10]], multioutput='raw_values')
    assert np.sqrt(mse) == pytest.approx(rmse)