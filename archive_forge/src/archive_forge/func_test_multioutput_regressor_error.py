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
def test_multioutput_regressor_error(pyplot):
    """Check that multioutput regressor raises correct error."""
    X = np.asarray([[0, 1], [1, 2]])
    y = np.asarray([[0, 1], [4, 1]])
    tree = DecisionTreeRegressor().fit(X, y)
    with pytest.raises(ValueError, match='Multi-output regressors are not supported'):
        DecisionBoundaryDisplay.from_estimator(tree, X, response_method='predict')