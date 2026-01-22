import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.metrics import check_scoring
@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
def test_warm_start_yields_identical_results(GradientBoosting, X, y):
    rng = 42
    gb_warm_start = GradientBoosting(n_iter_no_change=100, max_iter=50, random_state=rng, warm_start=True)
    gb_warm_start.fit(X, y).set_params(max_iter=75).fit(X, y)
    gb_no_warm_start = GradientBoosting(n_iter_no_change=100, max_iter=75, random_state=rng, warm_start=False)
    gb_no_warm_start.fit(X, y)
    _assert_predictor_equal(gb_warm_start, gb_no_warm_start, X)