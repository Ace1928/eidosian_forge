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
@pytest.mark.parametrize('SGDEstimator', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor, SGDOneClassSVM, SparseSGDOneClassSVM])
@pytest.mark.parametrize('data_type', (np.float32, np.float64))
def test_sgd_dtype_match(SGDEstimator, data_type):
    _X = X.astype(data_type)
    _Y = np.array(Y, dtype=data_type)
    sgd_model = SGDEstimator()
    sgd_model.fit(_X, _Y)
    assert sgd_model.coef_.dtype == data_type