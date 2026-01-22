import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg, optimize
from sklearn._loss.loss import (
from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSR_CONTAINERS
def random_X_y_coef(linear_model_loss, n_samples, n_features, coef_bound=(-2, 2), seed=42):
    """Random generate y, X and coef in valid range."""
    rng = np.random.RandomState(seed)
    n_dof = n_features + linear_model_loss.fit_intercept
    X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, random_state=rng)
    coef = linear_model_loss.init_zero_coef(X)
    if linear_model_loss.base_loss.is_multiclass:
        n_classes = linear_model_loss.base_loss.n_classes
        coef.flat[:] = rng.uniform(low=coef_bound[0], high=coef_bound[1], size=n_classes * n_dof)
        if linear_model_loss.fit_intercept:
            raw_prediction = X @ coef[:, :-1].T + coef[:, -1]
        else:
            raw_prediction = X @ coef.T
        proba = linear_model_loss.base_loss.link.inverse(raw_prediction)

        def choice_vectorized(items, p):
            s = p.cumsum(axis=1)
            r = rng.rand(p.shape[0])[:, None]
            k = (s < r).sum(axis=1)
            return items[k]
        y = choice_vectorized(np.arange(n_classes), p=proba).astype(np.float64)
    else:
        coef.flat[:] = rng.uniform(low=coef_bound[0], high=coef_bound[1], size=n_dof)
        if linear_model_loss.fit_intercept:
            raw_prediction = X @ coef[:-1] + coef[-1]
        else:
            raw_prediction = X @ coef
        y = linear_model_loss.base_loss.link.inverse(raw_prediction + rng.uniform(low=-1, high=1, size=n_samples))
    return (X, y, coef)