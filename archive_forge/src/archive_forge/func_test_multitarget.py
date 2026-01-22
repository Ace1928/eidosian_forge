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
@ignore_warnings
def test_multitarget():
    Y = np.vstack([y, y ** 2]).T
    n_targets = Y.shape[1]
    estimators = [linear_model.LassoLars(), linear_model.Lars(), linear_model.LassoLars(fit_intercept=False), linear_model.Lars(fit_intercept=False)]
    for estimator in estimators:
        estimator.fit(X, Y)
        Y_pred = estimator.predict(X)
        alphas, active, coef, path = (estimator.alphas_, estimator.active_, estimator.coef_, estimator.coef_path_)
        for k in range(n_targets):
            estimator.fit(X, Y[:, k])
            y_pred = estimator.predict(X)
            assert_array_almost_equal(alphas[k], estimator.alphas_)
            assert_array_almost_equal(active[k], estimator.active_)
            assert_array_almost_equal(coef[k], estimator.coef_)
            assert_array_almost_equal(path[k], estimator.coef_path_)
            assert_array_almost_equal(Y_pred[:, k], y_pred)