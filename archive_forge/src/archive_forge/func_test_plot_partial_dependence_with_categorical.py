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
@pytest.mark.parametrize('categorical_features, array_type', [(['col_A', 'col_C'], 'dataframe'), ([0, 2], 'array'), ([True, False, True], 'array')])
def test_plot_partial_dependence_with_categorical(pyplot, categorical_features, array_type):
    X = [[1, 1, 'A'], [2, 0, 'C'], [3, 2, 'B']]
    column_name = ['col_A', 'col_B', 'col_C']
    X = _convert_container(X, array_type, columns_name=column_name)
    y = np.array([1.2, 0.5, 0.45]).T
    preprocessor = make_column_transformer((OneHotEncoder(), categorical_features))
    model = make_pipeline(preprocessor, LinearRegression())
    model.fit(X, y)
    disp = PartialDependenceDisplay.from_estimator(model, X, features=['col_C'], feature_names=column_name, categorical_features=categorical_features)
    assert disp.figure_ is pyplot.gcf()
    assert disp.bars_.shape == (1, 1)
    assert disp.bars_[0][0] is not None
    assert disp.lines_.shape == (1, 1)
    assert disp.lines_[0][0] is None
    assert disp.contours_.shape == (1, 1)
    assert disp.contours_[0][0] is None
    assert disp.deciles_vlines_.shape == (1, 1)
    assert disp.deciles_vlines_[0][0] is None
    assert disp.deciles_hlines_.shape == (1, 1)
    assert disp.deciles_hlines_[0][0] is None
    assert disp.axes_[0, 0].get_legend() is None
    disp = PartialDependenceDisplay.from_estimator(model, X, features=[('col_A', 'col_C')], feature_names=column_name, categorical_features=categorical_features)
    assert disp.figure_ is pyplot.gcf()
    assert disp.bars_.shape == (1, 1)
    assert disp.bars_[0][0] is None
    assert disp.lines_.shape == (1, 1)
    assert disp.lines_[0][0] is None
    assert disp.contours_.shape == (1, 1)
    assert disp.contours_[0][0] is None
    assert disp.deciles_vlines_.shape == (1, 1)
    assert disp.deciles_vlines_[0][0] is None
    assert disp.deciles_hlines_.shape == (1, 1)
    assert disp.deciles_hlines_[0][0] is None
    assert disp.axes_[0, 0].get_legend() is None