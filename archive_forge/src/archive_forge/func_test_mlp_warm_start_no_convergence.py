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
@pytest.mark.parametrize('solver', ['sgd', 'adam', 'lbfgs'])
def test_mlp_warm_start_no_convergence(MLPEstimator, solver):
    """Check that we stop the number of iteration at `max_iter` when warm starting.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/24764
    """
    model = MLPEstimator(solver=solver, warm_start=True, early_stopping=False, max_iter=10, n_iter_no_change=np.inf, random_state=0)
    with pytest.warns(ConvergenceWarning):
        model.fit(X_iris, y_iris)
    assert model.n_iter_ == 10
    model.set_params(max_iter=20)
    with pytest.warns(ConvergenceWarning):
        model.fit(X_iris, y_iris)
    assert model.n_iter_ == 20