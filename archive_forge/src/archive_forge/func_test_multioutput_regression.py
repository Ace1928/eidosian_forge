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
def test_multioutput_regression():
    X, y = make_regression(n_samples=200, n_targets=5)
    mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=50, max_iter=200, random_state=1)
    mlp.fit(X, y)
    assert mlp.score(X, y) > 0.9