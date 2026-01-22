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
def test_ridge_loo_cv_asym_scoring():
    scoring = 'explained_variance'
    n_samples, n_features = (10, 5)
    n_targets = 1
    X, y = _make_sparse_offset_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, random_state=0, shuffle=False, noise=1, n_informative=5)
    alphas = [0.001, 0.1, 1.0, 10.0, 1000.0]
    loo_ridge = RidgeCV(cv=n_samples, fit_intercept=True, alphas=alphas, scoring=scoring)
    gcv_ridge = RidgeCV(fit_intercept=True, alphas=alphas, scoring=scoring)
    loo_ridge.fit(X, y)
    gcv_ridge.fit(X, y)
    assert gcv_ridge.alpha_ == pytest.approx(loo_ridge.alpha_)
    assert_allclose(gcv_ridge.coef_, loo_ridge.coef_, rtol=0.001)
    assert_allclose(gcv_ridge.intercept_, loo_ridge.intercept_, rtol=0.001)