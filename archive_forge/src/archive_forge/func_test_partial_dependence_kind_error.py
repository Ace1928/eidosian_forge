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
@pytest.mark.parametrize('features, kind', [([0, 2, (1, 2)], 'individual'), ([0, 2, (1, 2)], 'both'), ([(0, 1), (0, 2), (1, 2)], 'individual'), ([(0, 1), (0, 2), (1, 2)], 'both'), ([0, 2, (1, 2)], ['individual', 'individual', 'individual']), ([0, 2, (1, 2)], ['both', 'both', 'both'])])
def test_partial_dependence_kind_error(pyplot, clf_diabetes, diabetes, features, kind):
    """Check that we raise an informative error when 2-way PD is requested
    together with 1-way PD/ICE"""
    warn_msg = "ICE plot cannot be rendered for 2-way feature interactions. 2-way feature interactions mandates PD plots using the 'average' kind"
    with pytest.raises(ValueError, match=warn_msg):
        PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, features=features, grid_resolution=20, kind=kind)