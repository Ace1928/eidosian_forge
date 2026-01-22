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
def test_partial_fit_errors():
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    with pytest.raises(ValueError):
        MLPClassifier(solver='sgd').partial_fit(X, y, classes=[2])
    assert not hasattr(MLPClassifier(solver='lbfgs'), 'partial_fit')