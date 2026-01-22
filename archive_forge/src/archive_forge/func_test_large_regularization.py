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
@pytest.mark.parametrize('penalty', ['l2', 'l1', 'elasticnet'])
def test_large_regularization(penalty):
    model = SGDClassifier(alpha=100000.0, learning_rate='constant', eta0=0.1, penalty=penalty, shuffle=False, tol=None, max_iter=6)
    with np.errstate(all='raise'):
        model.fit(iris.data, iris.target)
    assert_array_almost_equal(model.coef_, np.zeros_like(model.coef_))