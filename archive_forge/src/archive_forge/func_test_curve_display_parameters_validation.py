import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import (
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('params, err_type, err_msg', [({'std_display_style': 'invalid'}, ValueError, 'Unknown std_display_style:'), ({'score_type': 'invalid'}, ValueError, 'Unknown score_type:')])
@pytest.mark.parametrize('CurveDisplay, specific_params', [(ValidationCurveDisplay, {'param_name': 'max_depth', 'param_range': [1, 3, 5]}), (LearningCurveDisplay, {'train_sizes': [0.3, 0.6, 0.9]})])
def test_curve_display_parameters_validation(pyplot, data, params, err_type, err_msg, CurveDisplay, specific_params):
    """Check that we raise a proper error when passing invalid parameters."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)
    with pytest.raises(err_type, match=err_msg):
        CurveDisplay.from_estimator(estimator, X, y, **specific_params, **params)