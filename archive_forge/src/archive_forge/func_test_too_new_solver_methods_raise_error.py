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
@pytest.mark.parametrize('solver', ('highs-ds', 'highs-ipm', 'highs'))
@pytest.mark.skipif(sp_version >= parse_version('1.6.0'), reason='Solvers are available as of scipy 1.6.0')
def test_too_new_solver_methods_raise_error(X_y_data, solver):
    """Test that highs solver raises for scipy<1.6.0."""
    X, y = X_y_data
    with pytest.raises(ValueError, match='scipy>=1.6.0'):
        QuantileRegressor(solver=solver).fit(X, y)