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
@pytest.mark.parametrize('nrows, ncols', [(2, 2), (3, 1)])
def test_plot_partial_dependence_incorrent_num_axes(pyplot, clf_diabetes, diabetes, nrows, ncols):
    grid_resolution = 5
    fig, axes = pyplot.subplots(nrows, ncols)
    axes_formats = [list(axes.ravel()), tuple(axes.ravel()), axes]
    msg = 'Expected ax to have 2 axes, got {}'.format(nrows * ncols)
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, ['age', 'bmi'], grid_resolution=grid_resolution, feature_names=diabetes.feature_names)
    for ax_format in axes_formats:
        with pytest.raises(ValueError, match=msg):
            PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, ['age', 'bmi'], grid_resolution=grid_resolution, feature_names=diabetes.feature_names, ax=ax_format)
        with pytest.raises(ValueError, match=msg):
            disp.plot(ax=ax_format)