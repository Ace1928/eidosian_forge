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
def test_dtype_match_cholesky():
    rng = np.random.RandomState(0)
    alpha = np.array([1.0, 0.5])
    n_samples, n_features, n_target = (6, 7, 2)
    X_64 = rng.randn(n_samples, n_features)
    y_64 = rng.randn(n_samples, n_target)
    X_32 = X_64.astype(np.float32)
    y_32 = y_64.astype(np.float32)
    ridge_32 = Ridge(alpha=alpha, solver='cholesky')
    ridge_32.fit(X_32, y_32)
    coef_32 = ridge_32.coef_
    ridge_64 = Ridge(alpha=alpha, solver='cholesky')
    ridge_64.fit(X_64, y_64)
    coef_64 = ridge_64.coef_
    assert coef_32.dtype == X_32.dtype
    assert coef_64.dtype == X_64.dtype
    assert ridge_32.predict(X_32).dtype == X_32.dtype
    assert ridge_64.predict(X_64).dtype == X_64.dtype
    assert_almost_equal(ridge_32.coef_, ridge_64.coef_, decimal=5)