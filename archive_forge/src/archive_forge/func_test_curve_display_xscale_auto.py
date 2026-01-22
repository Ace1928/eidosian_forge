import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import (
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('CurveDisplay, specific_params, expected_xscale', [(ValidationCurveDisplay, {'param_name': 'max_depth', 'param_range': np.arange(1, 5)}, 'linear'), (LearningCurveDisplay, {'train_sizes': np.linspace(0.1, 0.9, num=5)}, 'linear'), (ValidationCurveDisplay, {'param_name': 'max_depth', 'param_range': np.round(np.logspace(0, 2, num=5)).astype(np.int64)}, 'log'), (LearningCurveDisplay, {'train_sizes': np.logspace(-1, 0, num=5)}, 'log')])
def test_curve_display_xscale_auto(pyplot, data, CurveDisplay, specific_params, expected_xscale):
    """Check the behaviour of the x-axis scaling depending on the data provided."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)
    display = CurveDisplay.from_estimator(estimator, X, y, **specific_params)
    assert display.ax_.get_xscale() == expected_xscale