import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose
from scipy import linalg
from scipy.optimize import minimize, root
from sklearn._loss import HalfBinomialLoss, HalfPoissonLoss, HalfTweedieLoss
from sklearn._loss.link import IdentityLink, LogLink
from sklearn.base import clone
from sklearn.datasets import make_low_rank_matrix, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._glm import _GeneralizedLinearRegressor
from sklearn.linear_model._glm._newton_solver import NewtonCholeskySolver
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.metrics import d2_tweedie_score, mean_poisson_deviance
from sklearn.model_selection import train_test_split
@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('estimator', [PoissonRegressor(), GammaRegressor(), TweedieRegressor(power=3.0), TweedieRegressor(power=0, link='log'), TweedieRegressor(power=1.5), TweedieRegressor(power=4.5)])
def test_glm_log_regression(solver, fit_intercept, estimator):
    """Test GLM regression with log link on a simple dataset."""
    coef = [0.2, -0.1]
    X = np.array([[0, 1, 2, 3, 4], [1, 1, 1, 1, 1]]).T
    y = np.exp(np.dot(X, coef))
    glm = clone(estimator).set_params(alpha=0, fit_intercept=fit_intercept, solver=solver, tol=1e-08)
    if fit_intercept:
        res = glm.fit(X[:, :-1], y)
        assert_allclose(res.coef_, coef[:-1], rtol=1e-06)
        assert_allclose(res.intercept_, coef[-1], rtol=1e-06)
    else:
        res = glm.fit(X, y)
        assert_allclose(res.coef_, coef, rtol=2e-06)