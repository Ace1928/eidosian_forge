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
def test_average_sparse(klass):
    eta = 0.001
    alpha = 0.01
    clf = klass(loss='squared_error', learning_rate='constant', eta0=eta, alpha=alpha, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    n_samples = Y3.shape[0]
    clf.partial_fit(X3[:int(n_samples / 2)][:], Y3[:int(n_samples / 2)])
    clf.partial_fit(X3[int(n_samples / 2):][:], Y3[int(n_samples / 2):])
    average_weights, average_intercept = asgd(klass, X3, Y3, eta, alpha)
    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=16)