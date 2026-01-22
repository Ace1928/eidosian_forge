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
@pytest.mark.parametrize('X,y', classification_datasets)
def test_lbfgs_classification(X, y):
    X_train = X[:150]
    y_train = y[:150]
    X_test = X[150:]
    expected_shape_dtype = (X_test.shape[0], y_train.dtype.kind)
    for activation in ACTIVATION_TYPES:
        mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=50, max_iter=150, shuffle=True, random_state=1, activation=activation)
        mlp.fit(X_train, y_train)
        y_predict = mlp.predict(X_test)
        assert mlp.score(X_train, y_train) > 0.95
        assert (y_predict.shape[0], y_predict.dtype.kind) == expected_shape_dtype