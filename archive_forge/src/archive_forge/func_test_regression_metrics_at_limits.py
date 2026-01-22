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
def test_regression_metrics_at_limits():
    assert_almost_equal(mean_squared_error([0.0], [0.0]), 0.0)
    assert_almost_equal(root_mean_squared_error([0.0], [0.0]), 0.0)
    assert_almost_equal(mean_squared_log_error([0.0], [0.0]), 0.0)
    assert_almost_equal(mean_absolute_error([0.0], [0.0]), 0.0)
    assert_almost_equal(mean_pinball_loss([0.0], [0.0]), 0.0)
    assert_almost_equal(mean_absolute_percentage_error([0.0], [0.0]), 0.0)
    assert_almost_equal(median_absolute_error([0.0], [0.0]), 0.0)
    assert_almost_equal(max_error([0.0], [0.0]), 0.0)
    assert_almost_equal(explained_variance_score([0.0], [0.0]), 1.0)
    assert_almost_equal(r2_score([0.0, 1], [0.0, 1]), 1.0)
    assert_almost_equal(d2_pinball_score([0.0, 1], [0.0, 1]), 1.0)
    for s in (r2_score, explained_variance_score):
        assert_almost_equal(s([0, 0], [1, -1]), 0.0)
        assert_almost_equal(s([0, 0], [1, -1], force_finite=False), -np.inf)
        assert_almost_equal(s([1, 1], [1, 1]), 1.0)
        assert_almost_equal(s([1, 1], [1, 1], force_finite=False), np.nan)
    msg = 'Mean Squared Logarithmic Error cannot be used when targets contain negative values.'
    with pytest.raises(ValueError, match=msg):
        mean_squared_log_error([-1.0], [-1.0])
    msg = 'Mean Squared Logarithmic Error cannot be used when targets contain negative values.'
    with pytest.raises(ValueError, match=msg):
        mean_squared_log_error([1.0, 2.0, 3.0], [1.0, -2.0, 3.0])
    msg = 'Mean Squared Logarithmic Error cannot be used when targets contain negative values.'
    with pytest.raises(ValueError, match=msg):
        mean_squared_log_error([1.0, -2.0, 3.0], [1.0, 2.0, 3.0])
    msg = 'Root Mean Squared Logarithmic Error cannot be used when targets contain negative values.'
    with pytest.raises(ValueError, match=msg):
        root_mean_squared_log_error([1.0, -2.0, 3.0], [1.0, 2.0, 3.0])
    power = -1.2
    assert_allclose(mean_tweedie_deviance([0], [1.0], power=power), 2 / (2 - power), rtol=0.001)
    msg = 'can only be used on strictly positive y_pred.'
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)
    assert_almost_equal(mean_tweedie_deviance([0.0], [0.0], power=0), 0.0, 2)
    power = 1.0
    msg = 'only be used on non-negative y and strictly positive y_pred.'
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)
    power = 1.5
    assert_allclose(mean_tweedie_deviance([0.0], [1.0], power=power), 2 / (2 - power))
    msg = 'only be used on non-negative y and strictly positive y_pred.'
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)
    power = 2.0
    assert_allclose(mean_tweedie_deviance([1.0], [1.0], power=power), 0.0, atol=1e-08)
    msg = 'can only be used on strictly positive y and y_pred.'
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)
    power = 3.0
    assert_allclose(mean_tweedie_deviance([1.0], [1.0], power=power), 0.0, atol=1e-08)
    msg = 'can only be used on strictly positive y and y_pred.'
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)