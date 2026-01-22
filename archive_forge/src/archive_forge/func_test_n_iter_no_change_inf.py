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
@ignore_warnings(category=ConvergenceWarning)
def test_n_iter_no_change_inf():
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]
    tol = 1000000000.0
    n_iter_no_change = np.inf
    max_iter = 3000
    clf = MLPClassifier(tol=tol, max_iter=max_iter, solver='sgd', n_iter_no_change=n_iter_no_change)
    clf.fit(X, y)
    assert clf.n_iter_ == max_iter
    assert clf._no_improvement_count == clf.n_iter_ - 1