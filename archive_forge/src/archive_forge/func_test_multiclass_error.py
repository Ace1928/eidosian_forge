import warnings
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import (
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
@pytest.mark.parametrize('response_method', ['predict_proba', 'decision_function'])
def test_multiclass_error(pyplot, response_method):
    """Check multiclass errors."""
    X, y = make_classification(n_classes=3, n_informative=3, random_state=0)
    X = X[:, [0, 1]]
    lr = LogisticRegression().fit(X, y)
    msg = "Multiclass classifiers are only supported when `response_method` is 'predict' or 'auto'"
    with pytest.raises(ValueError, match=msg):
        DecisionBoundaryDisplay.from_estimator(lr, X, response_method=response_method)