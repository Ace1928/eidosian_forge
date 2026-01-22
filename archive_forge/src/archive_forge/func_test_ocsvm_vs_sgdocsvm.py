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
def test_ocsvm_vs_sgdocsvm():
    nu = 0.05
    gamma = 2.0
    random_state = 42
    rng = np.random.RandomState(random_state)
    X = 0.3 * rng.randn(500, 2)
    X_train = np.r_[X + 2, X - 2]
    X = 0.3 * rng.randn(100, 2)
    X_test = np.r_[X + 2, X - 2]
    clf = OneClassSVM(gamma=gamma, kernel='rbf', nu=nu)
    clf.fit(X_train)
    y_pred_ocsvm = clf.predict(X_test)
    dec_ocsvm = clf.decision_function(X_test).reshape(1, -1)
    max_iter = 15
    transform = Nystroem(gamma=gamma, random_state=random_state)
    clf_sgd = SGDOneClassSVM(nu=nu, shuffle=True, fit_intercept=True, max_iter=max_iter, random_state=random_state, tol=None)
    pipe_sgd = make_pipeline(transform, clf_sgd)
    pipe_sgd.fit(X_train)
    y_pred_sgdocsvm = pipe_sgd.predict(X_test)
    dec_sgdocsvm = pipe_sgd.decision_function(X_test).reshape(1, -1)
    assert np.mean(y_pred_sgdocsvm == y_pred_ocsvm) >= 0.99
    corrcoef = np.corrcoef(np.concatenate((dec_ocsvm, dec_sgdocsvm)))[0, 1]
    assert corrcoef >= 0.9