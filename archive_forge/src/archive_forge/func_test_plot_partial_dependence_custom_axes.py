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
def test_plot_partial_dependence_custom_axes(pyplot, clf_diabetes, diabetes):
    grid_resolution = 25
    fig, (ax1, ax2) = pyplot.subplots(1, 2)
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, ['age', ('age', 'bmi')], grid_resolution=grid_resolution, feature_names=diabetes.feature_names, ax=[ax1, ax2])
    assert fig is disp.figure_
    assert disp.bounding_ax_ is None
    assert disp.axes_.shape == (2,)
    assert disp.axes_[0] is ax1
    assert disp.axes_[1] is ax2
    ax = disp.axes_[0]
    assert ax.get_xlabel() == 'age'
    assert ax.get_ylabel() == 'Partial dependence'
    line = disp.lines_[0]
    avg_preds = disp.pd_results[0]
    target_idx = disp.target_idx
    line_data = line.get_data()
    assert_allclose(line_data[0], avg_preds['grid_values'][0])
    assert_allclose(line_data[1], avg_preds.average[target_idx].ravel())
    ax = disp.axes_[1]
    assert ax.get_xlabel() == 'age'
    assert ax.get_ylabel() == 'bmi'