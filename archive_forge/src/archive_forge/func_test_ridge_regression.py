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
@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_ridge_regression(solver, fit_intercept, ols_ridge_dataset, global_random_seed):
    """Test that Ridge converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    """
    X, y, _, coef = ols_ridge_dataset
    alpha = 1.0
    params = dict(alpha=alpha, fit_intercept=True, solver=solver, tol=1e-15 if solver in ('sag', 'saga') else 1e-10, random_state=global_random_seed)
    res_null = y - np.mean(y)
    res_Ridge = y - X @ coef
    R2_Ridge = 1 - np.sum(res_Ridge ** 2) / np.sum(res_null ** 2)
    model = Ridge(**params)
    X = X[:, :-1]
    if fit_intercept:
        intercept = coef[-1]
    else:
        X = X - X.mean(axis=0)
        y = y - y.mean()
        intercept = 0
    model.fit(X, y)
    coef = coef[:-1]
    assert model.intercept_ == pytest.approx(intercept)
    assert_allclose(model.coef_, coef)
    assert model.score(X, y) == pytest.approx(R2_Ridge)
    model = Ridge(**params).fit(X, y, sample_weight=np.ones(X.shape[0]))
    assert model.intercept_ == pytest.approx(intercept)
    assert_allclose(model.coef_, coef)
    assert model.score(X, y) == pytest.approx(R2_Ridge)