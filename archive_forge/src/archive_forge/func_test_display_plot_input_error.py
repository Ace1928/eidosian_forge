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
def test_display_plot_input_error(pyplot, fitted_clf):
    """Check input validation for `plot`."""
    disp = DecisionBoundaryDisplay.from_estimator(fitted_clf, X, grid_resolution=5)
    with pytest.raises(ValueError, match="plot_method must be 'contourf'"):
        disp.plot(plot_method='hello_world')