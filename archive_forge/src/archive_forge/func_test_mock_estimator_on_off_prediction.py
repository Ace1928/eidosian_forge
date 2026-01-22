import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._mocking import (
from sklearn.utils._testing import _convert_container
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('response_methods', [['predict'], ['predict', 'predict_proba'], ['predict', 'decision_function'], ['predict', 'predict_proba', 'decision_function']])
def test_mock_estimator_on_off_prediction(iris, response_methods):
    X, y = iris
    estimator = _MockEstimatorOnOffPrediction(response_methods=response_methods)
    estimator.fit(X, y)
    assert hasattr(estimator, 'classes_')
    assert_array_equal(estimator.classes_, np.unique(y))
    possible_responses = ['predict', 'predict_proba', 'decision_function']
    for response in possible_responses:
        if response in response_methods:
            assert hasattr(estimator, response)
            assert getattr(estimator, response)(X) == response
        else:
            assert not hasattr(estimator, response)