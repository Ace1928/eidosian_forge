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
@pytest.mark.parametrize('shape', [(10, 1), (13, 9), (3, 7), (2, 2), (20, 20)])
@pytest.mark.parametrize('uniform_weights', [True, False])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_compute_covariance(shape, uniform_weights, csr_container):
    rng = np.random.RandomState(0)
    X = rng.randn(*shape)
    if uniform_weights:
        sw = np.ones(X.shape[0])
    else:
        sw = rng.chisquare(1, shape[0])
    sqrt_sw = np.sqrt(sw)
    X_mean = np.average(X, axis=0, weights=sw)
    X_centered = (X - X_mean) * sqrt_sw[:, None]
    true_covariance = X_centered.T.dot(X_centered)
    X_sparse = csr_container(X * sqrt_sw[:, None])
    gcv = _RidgeGCV(fit_intercept=True)
    computed_cov, computed_mean = gcv._compute_covariance(X_sparse, sqrt_sw)
    assert_allclose(X_mean, computed_mean)
    assert_allclose(true_covariance, computed_cov)