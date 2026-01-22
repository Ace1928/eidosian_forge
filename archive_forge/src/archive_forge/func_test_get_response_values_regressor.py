import numpy as np
import pytest
from sklearn.datasets import (
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import (
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import _MockEstimatorOnOffPrediction
from sklearn.utils._response import _get_response_values, _get_response_values_binary
from sklearn.utils._testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('return_response_method_used', [True, False])
def test_get_response_values_regressor(return_response_method_used):
    """Check the behaviour of `_get_response_values` with regressor."""
    X, y = make_regression(n_samples=10, random_state=0)
    regressor = LinearRegression().fit(X, y)
    results = _get_response_values(regressor, X, response_method='predict', return_response_method_used=return_response_method_used)
    assert_array_equal(results[0], regressor.predict(X))
    assert results[1] is None
    if return_response_method_used:
        assert results[2] == 'predict'