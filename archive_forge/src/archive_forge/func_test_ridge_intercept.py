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
def test_ridge_intercept():
    rng = np.random.RandomState(0)
    n_samples, n_features = (5, 10)
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    Y = np.c_[y, 1.0 + y]
    ridge = Ridge()
    ridge.fit(X, y)
    intercept = ridge.intercept_
    ridge.fit(X, Y)
    assert_almost_equal(ridge.intercept_[0], intercept)
    assert_almost_equal(ridge.intercept_[1], intercept + 1.0)