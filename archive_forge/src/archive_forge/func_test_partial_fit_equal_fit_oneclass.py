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
@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
@pytest.mark.parametrize('lr', ['constant', 'optimal', 'invscaling', 'adaptive'])
def test_partial_fit_equal_fit_oneclass(klass, lr):
    clf = klass(nu=0.05, max_iter=2, eta0=0.01, learning_rate=lr, shuffle=False)
    clf.fit(X)
    y_scores = clf.decision_function(T)
    t = clf.t_
    coef = clf.coef_
    offset = clf.offset_
    clf = klass(nu=0.05, eta0=0.01, max_iter=1, learning_rate=lr, shuffle=False)
    for _ in range(2):
        clf.partial_fit(X)
    y_scores2 = clf.decision_function(T)
    assert clf.t_ == t
    assert_allclose(y_scores, y_scores2)
    assert_allclose(clf.coef_, coef)
    assert_allclose(clf.offset_, offset)