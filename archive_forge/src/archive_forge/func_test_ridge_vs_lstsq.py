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
def test_ridge_vs_lstsq():
    rng = np.random.RandomState(0)
    n_samples, n_features = (5, 4)
    y = rng.randn(n_samples)
    X = rng.randn(n_samples, n_features)
    ridge = Ridge(alpha=0.0, fit_intercept=False)
    ols = LinearRegression(fit_intercept=False)
    ridge.fit(X, y)
    ols.fit(X, y)
    assert_almost_equal(ridge.coef_, ols.coef_)
    ridge.fit(X, y)
    ols.fit(X, y)
    assert_almost_equal(ridge.coef_, ols.coef_)