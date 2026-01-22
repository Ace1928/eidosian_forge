import pickle
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils._testing import (
@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_sgd_least_squares_fit(klass):
    xmin, xmax = (-5, 5)
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)
    y = 0.5 * X.ravel()
    clf = klass(loss='squared_error', alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.99
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()
    clf = klass(loss='squared_error', alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.5