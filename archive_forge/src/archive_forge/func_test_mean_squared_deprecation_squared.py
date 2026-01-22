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
@pytest.mark.parametrize('metric', [mean_squared_error, mean_squared_log_error])
def test_mean_squared_deprecation_squared(metric):
    """Check the deprecation warning of the squared parameter"""
    depr_msg = "'squared' is deprecated in version 1.4 and will be removed in 1.6."
    y_true, y_pred = (np.arange(10), np.arange(1, 11))
    with pytest.warns(FutureWarning, match=depr_msg):
        metric(y_true, y_pred, squared=False)