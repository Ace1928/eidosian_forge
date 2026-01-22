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
def test_validation_mask_correctly_subsets(monkeypatch):
    """Test that data passed to validation callback correctly subsets.

    Non-regression test for #23255.
    """
    X, Y = (iris.data, iris.target)
    n_samples = X.shape[0]
    validation_fraction = 0.2
    clf = linear_model.SGDClassifier(early_stopping=True, tol=0.001, max_iter=1000, validation_fraction=validation_fraction)
    mock = Mock(side_effect=_stochastic_gradient._ValidationScoreCallback)
    monkeypatch.setattr(_stochastic_gradient, '_ValidationScoreCallback', mock)
    clf.fit(X, Y)
    X_val, y_val = mock.call_args[0][1:3]
    assert X_val.shape[0] == int(n_samples * validation_fraction)
    assert y_val.shape[0] == int(n_samples * validation_fraction)