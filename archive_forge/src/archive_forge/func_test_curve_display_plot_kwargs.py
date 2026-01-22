import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import (
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('CurveDisplay, specific_params', [(ValidationCurveDisplay, {'param_name': 'max_depth', 'param_range': [1, 3, 5]}), (LearningCurveDisplay, {'train_sizes': [0.3, 0.6, 0.9]})])
def test_curve_display_plot_kwargs(pyplot, data, CurveDisplay, specific_params):
    """Check the behaviour of the different plotting keyword arguments: `line_kw`,
    `fill_between_kw`, and `errorbar_kw`."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)
    std_display_style = 'fill_between'
    line_kw = {'color': 'red'}
    fill_between_kw = {'color': 'red', 'alpha': 1.0}
    display = CurveDisplay.from_estimator(estimator, X, y, **specific_params, std_display_style=std_display_style, line_kw=line_kw, fill_between_kw=fill_between_kw)
    assert display.lines_[0].get_color() == 'red'
    assert_allclose(display.fill_between_[0].get_facecolor(), [[1.0, 0.0, 0.0, 1.0]])
    std_display_style = 'errorbar'
    errorbar_kw = {'color': 'red'}
    display = CurveDisplay.from_estimator(estimator, X, y, **specific_params, std_display_style=std_display_style, errorbar_kw=errorbar_kw)
    assert display.errorbar_[0].lines[0].get_color() == 'red'