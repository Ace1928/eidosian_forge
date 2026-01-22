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
@pytest.mark.parametrize('klass, fit_params', [(SGDClassifier, {'intercept_init': 0}), (SparseSGDClassifier, {'intercept_init': 0}), (SGDOneClassSVM, {'offset_init': 0}), (SparseSGDOneClassSVM, {'offset_init': 0})])
def test_set_intercept_offset_binary(klass, fit_params):
    """Check that we can pass a scaler with binary classification to
    `intercept_init` or `offset_init`."""
    klass().fit(X5, Y5, **fit_params)