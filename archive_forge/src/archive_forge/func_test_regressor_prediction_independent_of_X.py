import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
@pytest.mark.parametrize('strategy', ['mean', 'median', 'quantile', 'constant'])
def test_regressor_prediction_independent_of_X(strategy):
    y = [0, 2, 1, 1]
    X1 = [[0]] * 4
    reg1 = DummyRegressor(strategy=strategy, constant=0, quantile=0.7)
    reg1.fit(X1, y)
    predictions1 = reg1.predict(X1)
    X2 = [[1]] * 4
    reg2 = DummyRegressor(strategy=strategy, constant=0, quantile=0.7)
    reg2.fit(X2, y)
    predictions2 = reg2.predict(X2)
    assert_array_equal(predictions1, predictions2)