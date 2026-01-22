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
@pytest.mark.skipif(parse_version(sp_version.base_version) >= parse_version('1.11'), reason='interior-point solver is not available in SciPy 1.11')
@pytest.mark.filterwarnings("ignore:`method='interior-point'` is deprecated")
def test_linprog_failure():
    """Test that linprog fails."""
    X = np.linspace(0, 10, num=10).reshape(-1, 1)
    y = np.linspace(0, 10, num=10)
    reg = QuantileRegressor(alpha=0, solver='interior-point', solver_options={'maxiter': 1})
    msg = 'Linear programming for QuantileRegressor did not succeed.'
    with pytest.warns(ConvergenceWarning, match=msg):
        reg.fit(X, y)