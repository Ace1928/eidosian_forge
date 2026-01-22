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
@pytest.fixture(params=['long', 'wide'])
def ols_ridge_dataset(global_random_seed, request):
    """Dataset with OLS and Ridge solutions, well conditioned X.

    The construction is based on the SVD decomposition of X = U S V'.

    Parameters
    ----------
    type : {"long", "wide"}
        If "long", then n_samples > n_features.
        If "wide", then n_features > n_samples.

    For "wide", we return the minimum norm solution w = X' (XX')^-1 y:

        min ||w||_2 subject to X w = y

    Returns
    -------
    X : ndarray
        Last column of 1, i.e. intercept.
    y : ndarray
    coef_ols : ndarray of shape
        Minimum norm OLS solutions, i.e. min ||X w - y||_2_2 (with minimum ||w||_2 in
        case of ambiguity)
        Last coefficient is intercept.
    coef_ridge : ndarray of shape (5,)
        Ridge solution with alpha=1, i.e. min ||X w - y||_2_2 + ||w||_2^2.
        Last coefficient is intercept.
    """
    if request.param == 'long':
        n_samples, n_features = (12, 4)
    else:
        n_samples, n_features = (4, 12)
    k = min(n_samples, n_features)
    rng = np.random.RandomState(global_random_seed)
    X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=k, random_state=rng)
    X[:, -1] = 1
    U, s, Vt = linalg.svd(X)
    assert np.all(s > 0.001)
    U1, U2 = (U[:, :k], U[:, k:])
    Vt1, _ = (Vt[:k, :], Vt[k:, :])
    if request.param == 'long':
        coef_ols = rng.uniform(low=-10, high=10, size=n_features)
        y = X @ coef_ols
        y += U2 @ rng.normal(size=n_samples - n_features) ** 2
    else:
        y = rng.uniform(low=-10, high=10, size=n_samples)
        coef_ols = Vt1.T @ np.diag(1 / s) @ U1.T @ y
    alpha = 1
    d = alpha * np.identity(n_features)
    d[-1, -1] = 0
    coef_ridge = linalg.solve(X.T @ X + d, X.T @ y)
    R_OLS = y - X @ coef_ols
    R_Ridge = y - X @ coef_ridge
    assert np.linalg.norm(R_OLS) < np.linalg.norm(R_Ridge)
    return (X, y, coef_ols, coef_ridge)