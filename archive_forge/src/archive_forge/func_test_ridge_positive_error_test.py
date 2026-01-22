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
@pytest.mark.parametrize('solver', ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
def test_ridge_positive_error_test(solver):
    """Test input validation for positive argument in Ridge."""
    alpha = 0.1
    X = np.array([[1, 2], [3, 4]])
    coef = np.array([1, -1])
    y = X @ coef
    model = Ridge(alpha=alpha, positive=True, solver=solver, fit_intercept=False)
    with pytest.raises(ValueError, match='does not support positive'):
        model.fit(X, y)
    with pytest.raises(ValueError, match="only 'lbfgs' solver can be used"):
        _, _ = ridge_regression(X, y, alpha, positive=True, solver=solver, return_intercept=False)