import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
def test_lasso_lars_vs_lasso_cd_early_stopping():
    alphas_min = [10, 0.9, 0.0001]
    X = diabetes.data
    for alpha_min in alphas_min:
        alphas, _, lasso_path = linear_model.lars_path(X, y, method='lasso', alpha_min=alpha_min)
        lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-08)
        lasso_cd.alpha = alphas[-1]
        lasso_cd.fit(X, y)
        error = linalg.norm(lasso_path[:, -1] - lasso_cd.coef_)
        assert error < 0.01
    X = diabetes.data - diabetes.data.sum(axis=0)
    X /= np.linalg.norm(X, axis=0)
    for alpha_min in alphas_min:
        alphas, _, lasso_path = linear_model.lars_path(X, y, method='lasso', alpha_min=alpha_min)
        lasso_cd = linear_model.Lasso(tol=1e-08)
        lasso_cd.alpha = alphas[-1]
        lasso_cd.fit(X, y)
        error = linalg.norm(lasso_path[:, -1] - lasso_cd.coef_)
        assert error < 0.01