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
@pytest.mark.parametrize('target', [0, 1])
def test_plot_partial_dependence_multioutput(pyplot, target):
    X, y = multioutput_regression_data
    clf = LinearRegression().fit(X, y)
    grid_resolution = 25
    disp = PartialDependenceDisplay.from_estimator(clf, X, [0, 1], target=target, grid_resolution=grid_resolution)
    fig = pyplot.gcf()
    axs = fig.get_axes()
    assert len(axs) == 3
    assert disp.target_idx == target
    assert disp.bounding_ax_ is not None
    positions = [(0, 0), (0, 1)]
    expected_label = ['Partial dependence', '']
    for i, pos in enumerate(positions):
        ax = disp.axes_[pos]
        assert ax.get_ylabel() == expected_label[i]
        assert ax.get_xlabel() == f'x{i}'