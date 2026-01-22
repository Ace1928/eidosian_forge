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
def test_ridge_cv_individual_penalties():
    rng = np.random.RandomState(42)
    n_samples, n_features, n_targets = (20, 5, 3)
    y = rng.randn(n_samples, n_targets)
    X = np.dot(y[:, [0]], np.ones((1, n_features))) + np.dot(y[:, [1]], 0.05 * np.ones((1, n_features))) + np.dot(y[:, [2]], 0.001 * np.ones((1, n_features))) + rng.randn(n_samples, n_features)
    alphas = (1, 100, 1000)
    optimal_alphas = [RidgeCV(alphas=alphas).fit(X, target).alpha_ for target in y.T]
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True).fit(X, y)
    assert_array_equal(optimal_alphas, ridge_cv.alpha_)
    assert_array_almost_equal(Ridge(alpha=ridge_cv.alpha_).fit(X, y).coef_, ridge_cv.coef_)
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True, store_cv_values=True).fit(X, y)
    assert ridge_cv.alpha_.shape == (n_targets,)
    assert ridge_cv.best_score_.shape == (n_targets,)
    assert ridge_cv.cv_values_.shape == (n_samples, len(alphas), n_targets)
    ridge_cv = RidgeCV(alphas=1, alpha_per_target=True, store_cv_values=True).fit(X, y)
    assert ridge_cv.alpha_.shape == (n_targets,)
    assert ridge_cv.best_score_.shape == (n_targets,)
    assert ridge_cv.cv_values_.shape == (n_samples, n_targets, 1)
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True, store_cv_values=True).fit(X, y[:, 0])
    assert np.isscalar(ridge_cv.alpha_)
    assert np.isscalar(ridge_cv.best_score_)
    assert ridge_cv.cv_values_.shape == (n_samples, len(alphas))
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True, scoring='r2').fit(X, y)
    assert_array_equal(optimal_alphas, ridge_cv.alpha_)
    assert_array_almost_equal(Ridge(alpha=ridge_cv.alpha_).fit(X, y).coef_, ridge_cv.coef_)
    ridge_cv = RidgeCV(alphas=alphas, cv=LeaveOneOut(), alpha_per_target=True)
    msg = 'cv!=None and alpha_per_target=True are incompatible'
    with pytest.raises(ValueError, match=msg):
        ridge_cv.fit(X, y)
    ridge_cv = RidgeCV(alphas=alphas, cv=6, alpha_per_target=True)
    with pytest.raises(ValueError, match=msg):
        ridge_cv.fit(X, y)