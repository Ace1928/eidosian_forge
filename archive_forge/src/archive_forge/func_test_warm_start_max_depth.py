import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.metrics import check_scoring
@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
def test_warm_start_max_depth(GradientBoosting, X, y):
    gb = GradientBoosting(max_iter=20, min_samples_leaf=1, warm_start=True, max_depth=2, early_stopping=False)
    gb.fit(X, y)
    gb.set_params(max_iter=30, max_depth=3, n_iter_no_change=110)
    gb.fit(X, y)
    for i in range(20):
        assert gb._predictors[i][0].get_max_depth() == 2
    for i in range(1, 11):
        assert gb._predictors[-i][0].get_max_depth() == 3