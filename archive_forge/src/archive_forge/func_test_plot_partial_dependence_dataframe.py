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
def test_plot_partial_dependence_dataframe(pyplot, clf_diabetes, diabetes):
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    grid_resolution = 25
    PartialDependenceDisplay.from_estimator(clf_diabetes, df, ['bp', 's1'], grid_resolution=grid_resolution, feature_names=df.columns.tolist())