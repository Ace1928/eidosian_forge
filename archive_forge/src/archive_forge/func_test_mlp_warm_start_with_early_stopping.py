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
def test_mlp_warm_start_with_early_stopping(MLPEstimator):
    """Check that early stopping works with warm start."""
    mlp = MLPEstimator(max_iter=10, random_state=0, warm_start=True, early_stopping=True)
    mlp.fit(X_iris, y_iris)
    n_validation_scores = len(mlp.validation_scores_)
    mlp.set_params(max_iter=20)
    mlp.fit(X_iris, y_iris)
    assert len(mlp.validation_scores_) > n_validation_scores