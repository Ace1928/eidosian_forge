import re
import sys
import warnings
from io import StringIO
import joblib
import numpy as np
import pytest
from numpy.testing import (
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, scale
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('MLPEstimator', [MLPClassifier, MLPRegressor])
def test_warm_start_full_iteration(MLPEstimator):
    X, y = (X_iris, y_iris)
    max_iter = 3
    clf = MLPEstimator(hidden_layer_sizes=2, solver='sgd', warm_start=True, max_iter=max_iter)
    clf.fit(X, y)
    assert max_iter == clf.n_iter_
    clf.fit(X, y)
    assert max_iter == clf.n_iter_