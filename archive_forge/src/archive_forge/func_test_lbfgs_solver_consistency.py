import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('alpha', [0.001, 0.01, 0.1, 1.0])
def test_lbfgs_solver_consistency(alpha):
    """Test that LBGFS gets almost the same coef of svd when positive=False."""
    X, y = make_regression(n_samples=300, n_features=300, random_state=42)
    y = np.expand_dims(y, 1)
    alpha = np.asarray([alpha])
    config = {'positive': False, 'tol': 1e-16, 'max_iter': 500000}
    coef_lbfgs = _solve_lbfgs(X, y, alpha, **config)
    coef_cholesky = _solve_svd(X, y, alpha)
    assert_allclose(coef_lbfgs, coef_cholesky, atol=0.0001, rtol=0)