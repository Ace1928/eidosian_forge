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
@pytest.mark.parametrize('X,y', regression_datasets)
def test_lbfgs_regression_maxfun(X, y):
    max_fun = 10
    for activation in ACTIVATION_TYPES:
        mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=50, tol=0.0, max_iter=150, max_fun=max_fun, shuffle=True, random_state=1, activation=activation)
        with pytest.warns(ConvergenceWarning):
            mlp.fit(X, y)
            assert max_fun >= mlp.n_iter_