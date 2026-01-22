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
def test_sgd_multiclass_average(klass):
    eta = 0.001
    alpha = 0.01
    clf = klass(loss='squared_error', learning_rate='constant', eta0=eta, alpha=alpha, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    np_Y2 = np.array(Y2)
    clf.fit(X2, np_Y2)
    classes = np.unique(np_Y2)
    for i, cl in enumerate(classes):
        y_i = np.ones(np_Y2.shape[0])
        y_i[np_Y2 != cl] = -1
        average_coef, average_intercept = asgd(klass, X2, y_i, eta, alpha)
        assert_array_almost_equal(average_coef, clf.coef_[i], decimal=16)
        assert_almost_equal(average_intercept, clf.intercept_[i], decimal=16)