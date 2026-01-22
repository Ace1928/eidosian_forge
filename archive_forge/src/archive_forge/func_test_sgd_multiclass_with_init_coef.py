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
def test_sgd_multiclass_with_init_coef(klass):
    clf = klass(alpha=0.01, max_iter=20)
    clf.fit(X2, Y2, coef_init=np.zeros((3, 2)), intercept_init=np.zeros(3))
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape, (3,)
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)