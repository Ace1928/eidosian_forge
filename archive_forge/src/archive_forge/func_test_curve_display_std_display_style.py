import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import (
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('CurveDisplay, specific_params', [(ValidationCurveDisplay, {'param_name': 'max_depth', 'param_range': [1, 3, 5]}), (LearningCurveDisplay, {'train_sizes': [0.3, 0.6, 0.9]})])
def test_curve_display_std_display_style(pyplot, data, CurveDisplay, specific_params):
    """Check the behaviour of the parameter `std_display_style`."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)
    import matplotlib as mpl
    std_display_style = None
    display = CurveDisplay.from_estimator(estimator, X, y, **specific_params, std_display_style=std_display_style)
    assert len(display.lines_) == 2
    for line in display.lines_:
        assert isinstance(line, mpl.lines.Line2D)
    assert display.errorbar_ is None
    assert display.fill_between_ is None
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert len(legend_label) == 2
    std_display_style = 'fill_between'
    display = CurveDisplay.from_estimator(estimator, X, y, **specific_params, std_display_style=std_display_style)
    assert len(display.lines_) == 2
    for line in display.lines_:
        assert isinstance(line, mpl.lines.Line2D)
    assert display.errorbar_ is None
    assert len(display.fill_between_) == 2
    for fill_between in display.fill_between_:
        assert isinstance(fill_between, mpl.collections.PolyCollection)
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert len(legend_label) == 2
    std_display_style = 'errorbar'
    display = CurveDisplay.from_estimator(estimator, X, y, **specific_params, std_display_style=std_display_style)
    assert display.lines_ is None
    assert len(display.errorbar_) == 2
    for errorbar in display.errorbar_:
        assert isinstance(errorbar, mpl.container.ErrorbarContainer)
    assert display.fill_between_ is None
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert len(legend_label) == 2