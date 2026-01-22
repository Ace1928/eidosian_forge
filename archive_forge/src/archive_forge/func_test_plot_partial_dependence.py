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
@pytest.mark.parametrize('grid_resolution', [10, 20])
def test_plot_partial_dependence(grid_resolution, pyplot, clf_diabetes, diabetes):
    feature_names = diabetes.feature_names
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, [0, 2, (0, 2)], grid_resolution=grid_resolution, feature_names=feature_names, contour_kw={'cmap': 'jet'})
    fig = pyplot.gcf()
    axs = fig.get_axes()
    assert disp.figure_ is fig
    assert len(axs) == 4
    assert disp.bounding_ax_ is not None
    assert disp.axes_.shape == (1, 3)
    assert disp.lines_.shape == (1, 3)
    assert disp.contours_.shape == (1, 3)
    assert disp.deciles_vlines_.shape == (1, 3)
    assert disp.deciles_hlines_.shape == (1, 3)
    assert disp.lines_[0, 2] is None
    assert disp.contours_[0, 0] is None
    assert disp.contours_[0, 1] is None
    for i in range(3):
        assert disp.deciles_vlines_[0, i] is not None
    assert disp.deciles_hlines_[0, 0] is None
    assert disp.deciles_hlines_[0, 1] is None
    assert disp.deciles_hlines_[0, 2] is not None
    assert disp.features == [(0,), (2,), (0, 2)]
    assert np.all(disp.feature_names == feature_names)
    assert len(disp.deciles) == 2
    for i in [0, 2]:
        assert_allclose(disp.deciles[i], mquantiles(diabetes.data[:, i], prob=np.arange(0.1, 1.0, 0.1)))
    single_feature_positions = [(0, (0, 0)), (2, (0, 1))]
    expected_ylabels = ['Partial dependence', '']
    for i, (feat_col, pos) in enumerate(single_feature_positions):
        ax = disp.axes_[pos]
        assert ax.get_ylabel() == expected_ylabels[i]
        assert ax.get_xlabel() == diabetes.feature_names[feat_col]
        line = disp.lines_[pos]
        avg_preds = disp.pd_results[i]
        assert avg_preds.average.shape == (1, grid_resolution)
        target_idx = disp.target_idx
        line_data = line.get_data()
        assert_allclose(line_data[0], avg_preds['grid_values'][0])
        assert_allclose(line_data[1], avg_preds.average[target_idx].ravel())
    ax = disp.axes_[0, 2]
    coutour = disp.contours_[0, 2]
    assert coutour.get_cmap().name == 'jet'
    assert ax.get_xlabel() == diabetes.feature_names[0]
    assert ax.get_ylabel() == diabetes.feature_names[2]