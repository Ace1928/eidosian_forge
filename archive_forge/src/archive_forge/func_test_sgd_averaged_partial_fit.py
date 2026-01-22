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
def test_sgd_averaged_partial_fit(klass):
    eta = 0.001
    alpha = 0.01
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    y = np.dot(X, w)
    clf = klass(loss='squared_error', learning_rate='constant', eta0=eta, alpha=alpha, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    clf.partial_fit(X[:int(n_samples / 2)][:], y[:int(n_samples / 2)])
    clf.partial_fit(X[int(n_samples / 2):][:], y[int(n_samples / 2):])
    average_weights, average_intercept = asgd(klass, X, y, eta, alpha)
    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_[0], average_intercept, decimal=16)