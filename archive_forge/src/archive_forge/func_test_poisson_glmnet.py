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
@pytest.mark.parametrize('solver', ['lbfgs', 'newton-cholesky'])
def test_poisson_glmnet(solver):
    """Compare Poisson regression with L2 regularization and LogLink to glmnet"""
    X = np.array([[-2, -1, 1, 2], [0, 0, 1, 1]]).T
    y = np.array([0, 1, 1, 2])
    glm = PoissonRegressor(alpha=1, fit_intercept=True, tol=1e-07, max_iter=300, solver=solver)
    glm.fit(X, y)
    assert_allclose(glm.intercept_, -0.12889386979, rtol=1e-05)
    assert_allclose(glm.coef_, [0.29019207995, 0.03741173122], rtol=1e-05)