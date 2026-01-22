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
def test_mean_absolute_percentage_error():
    random_number_generator = np.random.RandomState(42)
    y_true = random_number_generator.exponential(size=100)
    y_pred = 1.2 * y_true
    assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(0.2)