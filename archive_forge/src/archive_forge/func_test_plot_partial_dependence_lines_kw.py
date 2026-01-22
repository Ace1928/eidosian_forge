import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats.mstats import mquantiles
from sklearn.compose import make_column_transformer
from sklearn.datasets import (
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import _convert_container
@pytest.mark.filterwarnings('ignore:A Bunch will be returned')
@pytest.mark.parametrize('line_kw, pd_line_kw, ice_lines_kw, expected_colors', [({'color': 'r'}, {'color': 'g'}, {'color': 'b'}, ('g', 'b')), (None, {'color': 'g'}, {'color': 'b'}, ('g', 'b')), ({'color': 'r'}, None, {'color': 'b'}, ('r', 'b')), ({'color': 'r'}, {'color': 'g'}, None, ('g', 'r')), ({'color': 'r'}, None, None, ('r', 'r')), ({'color': 'r'}, {'linestyle': '--'}, {'linestyle': '-.'}, ('r', 'r'))])
def test_plot_partial_dependence_lines_kw(pyplot, clf_diabetes, diabetes, line_kw, pd_line_kw, ice_lines_kw, expected_colors):
    """Check that passing `pd_line_kw` and `ice_lines_kw` will act on the
    specific lines in the plot.
    """
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, [0, 2], grid_resolution=20, feature_names=diabetes.feature_names, n_cols=2, kind='both', line_kw=line_kw, pd_line_kw=pd_line_kw, ice_lines_kw=ice_lines_kw)
    line = disp.lines_[0, 0, -1]
    assert line.get_color() == expected_colors[0]
    if pd_line_kw is not None and 'linestyle' in pd_line_kw:
        assert line.get_linestyle() == pd_line_kw['linestyle']
    else:
        assert line.get_linestyle() == '--'
    line = disp.lines_[0, 0, 0]
    assert line.get_color() == expected_colors[1]
    if ice_lines_kw is not None and 'linestyle' in ice_lines_kw:
        assert line.get_linestyle() == ice_lines_kw['linestyle']
    else:
        assert line.get_linestyle() == '-'