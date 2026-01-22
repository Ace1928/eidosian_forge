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
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('Estimator', [MLPClassifier, MLPRegressor])
def test_mlp_param_dtypes(dtype, Estimator):
    X, y = (X_digits.astype(dtype), y_digits)
    mlp = Estimator(alpha=1e-05, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50)
    mlp.fit(X[:300], y[:300])
    pred = mlp.predict(X[300:])
    assert all([intercept.dtype == dtype for intercept in mlp.intercepts_])
    assert all([coef.dtype == dtype for coef in mlp.coefs_])
    if Estimator == MLPRegressor:
        assert pred.dtype == dtype