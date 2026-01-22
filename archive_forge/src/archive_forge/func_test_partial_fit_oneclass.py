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
def test_partial_fit_oneclass(klass):
    third = X.shape[0] // 3
    clf = klass(nu=0.1)
    clf.partial_fit(X[:third])
    assert clf.coef_.shape == (X.shape[1],)
    assert clf.offset_.shape == (1,)
    assert clf.predict([[0, 0]]).shape == (1,)
    previous_coefs = clf.coef_
    clf.partial_fit(X[third:])
    assert clf.coef_ is previous_coefs
    with pytest.raises(ValueError):
        clf.partial_fit(X[:, 1])