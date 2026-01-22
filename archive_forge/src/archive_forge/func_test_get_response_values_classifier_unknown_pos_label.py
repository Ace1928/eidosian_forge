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
@pytest.mark.parametrize('response_method', ['predict_proba', 'decision_function', 'predict', 'predict_log_proba'])
def test_get_response_values_classifier_unknown_pos_label(response_method):
    """Check that `_get_response_values` raises the proper error message with
    classifier."""
    X, y = make_classification(n_samples=10, n_classes=2, random_state=0)
    classifier = LogisticRegression().fit(X, y)
    err_msg = 'pos_label=whatever is not a valid label: It should be one of \\[0 1\\]'
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values(classifier, X, response_method=response_method, pos_label='whatever')