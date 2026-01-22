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
@pytest.mark.parametrize('kind, line_kw, label', [('individual', {}, None), ('individual', {'label': 'xxx'}, None), ('average', {}, None), ('average', {'label': 'xxx'}, 'xxx'), ('both', {}, 'average'), ('both', {'label': 'xxx'}, 'xxx')])
def test_partial_dependence_overwrite_labels(pyplot, clf_diabetes, diabetes, kind, line_kw, label):
    """Test that make sure that we can overwrite the label of the PDP plot"""
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, [0, 2], grid_resolution=25, feature_names=diabetes.feature_names, kind=kind, line_kw=line_kw)
    for ax in disp.axes_.ravel():
        if label is None:
            assert ax.get_legend() is None
        else:
            legend_text = ax.get_legend().get_texts()
            assert len(legend_text) == 1
            assert legend_text[0].get_text() == label