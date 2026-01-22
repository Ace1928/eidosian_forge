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
@pytest.mark.parametrize('kwargs, error_msg', [({'plot_method': 'hello_world'}, 'plot_method must be one of contourf, contour, pcolormesh. Got hello_world instead.'), ({'grid_resolution': 1}, 'grid_resolution must be greater than 1. Got 1 instead'), ({'grid_resolution': -1}, 'grid_resolution must be greater than 1. Got -1 instead'), ({'eps': -1.1}, 'eps must be greater than or equal to 0. Got -1.1 instead')])
def test_input_validation_errors(pyplot, kwargs, error_msg, fitted_clf):
    """Check input validation from_estimator."""
    with pytest.raises(ValueError, match=error_msg):
        DecisionBoundaryDisplay.from_estimator(fitted_clf, X, **kwargs)