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
def test_regression_losses(klass):
    random_state = np.random.RandomState(1)
    clf = klass(alpha=0.01, learning_rate='constant', eta0=0.1, loss='epsilon_insensitive', random_state=random_state)
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)
    clf = klass(alpha=0.01, learning_rate='constant', eta0=0.1, loss='squared_epsilon_insensitive', random_state=random_state)
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)
    clf = klass(alpha=0.01, loss='huber', random_state=random_state)
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)
    clf = klass(alpha=0.01, learning_rate='constant', eta0=0.01, loss='squared_error', random_state=random_state)
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)