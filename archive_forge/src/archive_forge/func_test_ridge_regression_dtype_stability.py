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
@pytest.mark.parametrize('solver', ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'])
@pytest.mark.parametrize('seed', range(1))
def test_ridge_regression_dtype_stability(solver, seed):
    random_state = np.random.RandomState(seed)
    n_samples, n_features = (6, 5)
    X = random_state.randn(n_samples, n_features)
    coef = random_state.randn(n_features)
    y = np.dot(X, coef) + 0.01 * random_state.randn(n_samples)
    alpha = 1.0
    positive = solver == 'lbfgs'
    results = dict()
    atol = 0.001 if solver == 'sparse_cg' else 1e-05
    for current_dtype in (np.float32, np.float64):
        results[current_dtype] = ridge_regression(X.astype(current_dtype), y.astype(current_dtype), alpha=alpha, solver=solver, random_state=random_state, sample_weight=None, positive=positive, max_iter=500, tol=1e-10, return_n_iter=False, return_intercept=False)
    assert results[np.float32].dtype == np.float32
    assert results[np.float64].dtype == np.float64
    assert_allclose(results[np.float32], results[np.float64], atol=atol)