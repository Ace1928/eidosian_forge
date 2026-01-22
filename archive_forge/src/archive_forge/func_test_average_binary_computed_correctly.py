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
@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_average_binary_computed_correctly(klass):
    eta = 0.1
    alpha = 2.0
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    clf = klass(loss='squared_error', learning_rate='constant', eta0=eta, alpha=alpha, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    y = np.dot(X, w)
    y = np.sign(y)
    clf.fit(X, y)
    average_weights, average_intercept = asgd(klass, X, y, eta, alpha)
    average_weights = average_weights.reshape(1, -1)
    assert_array_almost_equal(clf.coef_, average_weights, decimal=14)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=14)