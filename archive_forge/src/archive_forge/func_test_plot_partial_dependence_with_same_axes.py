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
def test_plot_partial_dependence_with_same_axes(pyplot, clf_diabetes, diabetes):
    grid_resolution = 25
    fig, ax = pyplot.subplots()
    PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, ['age', 'bmi'], grid_resolution=grid_resolution, feature_names=diabetes.feature_names, ax=ax)
    msg = 'The ax was already used in another plot function, please set ax=display.axes_ instead'
    with pytest.raises(ValueError, match=msg):
        PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, ['age', 'bmi'], grid_resolution=grid_resolution, feature_names=diabetes.feature_names, ax=ax)