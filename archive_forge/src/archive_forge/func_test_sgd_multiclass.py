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
def test_sgd_multiclass(klass):
    clf = klass(alpha=0.01, max_iter=20).fit(X2, Y2)
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape == (3,)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)