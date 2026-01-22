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
@pytest.mark.parametrize('solver, sparse_container', ((solver, sparse_container) for solver, sparse_container in product(['cholesky', 'sag', 'sparse_cg', 'lsqr', 'saga', 'ridgecv'], [None] + CSR_CONTAINERS) if sparse_container is None or solver in ['sparse_cg', 'ridgecv']))
@pytest.mark.parametrize('n_samples,dtype,proportion_nonzero', [(20, 'float32', 0.1), (40, 'float32', 1.0), (20, 'float64', 0.2)])
@pytest.mark.parametrize('seed', np.arange(3))
def test_solver_consistency(solver, proportion_nonzero, n_samples, dtype, sparse_container, seed):
    alpha = 1.0
    noise = 50.0 if proportion_nonzero > 0.9 else 500.0
    X, y = _make_sparse_offset_regression(bias=10, n_features=30, proportion_nonzero=proportion_nonzero, noise=noise, random_state=seed, n_samples=n_samples)
    X = minmax_scale(X)
    svd_ridge = Ridge(solver='svd', alpha=alpha).fit(X, y)
    X = X.astype(dtype, copy=False)
    y = y.astype(dtype, copy=False)
    if sparse_container is not None:
        X = sparse_container(X)
    if solver == 'ridgecv':
        ridge = RidgeCV(alphas=[alpha])
    else:
        ridge = Ridge(solver=solver, tol=1e-10, alpha=alpha)
    ridge.fit(X, y)
    assert_allclose(ridge.coef_, svd_ridge.coef_, atol=0.001, rtol=0.001)
    assert_allclose(ridge.intercept_, svd_ridge.intercept_, atol=0.001, rtol=0.001)