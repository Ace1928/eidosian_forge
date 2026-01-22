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
def test_plot_partial_dependence_does_not_override_ylabel(pyplot, clf_diabetes, diabetes):
    _, axes = pyplot.subplots(1, 2)
    axes[0].set_ylabel('Hello world')
    PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, [0, 1], ax=axes)
    assert axes[0].get_ylabel() == 'Hello world'
    assert axes[1].get_ylabel() == 'Partial dependence'