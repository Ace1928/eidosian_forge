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
@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDOneClassSVM, SparseSGDOneClassSVM])
def test_wrong_sample_weights(klass):
    if klass in [SGDClassifier, SparseSGDClassifier]:
        clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False)
    elif klass in [SGDOneClassSVM, SparseSGDOneClassSVM]:
        clf = klass(nu=0.1, max_iter=1000, fit_intercept=False)
    with pytest.raises(ValueError):
        clf.fit(X, Y, sample_weight=np.arange(7))