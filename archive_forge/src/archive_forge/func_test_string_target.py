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
def test_string_target(pyplot):
    """Check that decision boundary works with classifiers trained on string labels."""
    iris = load_iris()
    X = iris.data[:, [0, 1]]
    y = iris.target_names[iris.target]
    log_reg = LogisticRegression().fit(X, y)
    DecisionBoundaryDisplay.from_estimator(log_reg, X, grid_resolution=5, response_method='predict')