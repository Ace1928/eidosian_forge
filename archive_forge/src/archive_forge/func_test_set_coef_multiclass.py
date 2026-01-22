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
def test_set_coef_multiclass(klass):
    clf = klass()
    with pytest.raises(ValueError):
        clf.fit(X2, Y2, coef_init=np.zeros((2, 2)))
    clf = klass().fit(X2, Y2, coef_init=np.zeros((3, 2)))
    clf = klass()
    with pytest.raises(ValueError):
        clf.fit(X2, Y2, intercept_init=np.zeros((1,)))
    clf = klass().fit(X2, Y2, intercept_init=np.zeros((3,)))