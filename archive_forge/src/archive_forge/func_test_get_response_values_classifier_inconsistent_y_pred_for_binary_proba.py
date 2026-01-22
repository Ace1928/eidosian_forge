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
@pytest.mark.parametrize('response_method', ['predict_proba', 'predict_log_proba'])
def test_get_response_values_classifier_inconsistent_y_pred_for_binary_proba(response_method):
    """Check that `_get_response_values` will raise an error when `y_pred` has a
    single class with `predict_proba`."""
    X, y_two_class = make_classification(n_samples=10, n_classes=2, random_state=0)
    y_single_class = np.zeros_like(y_two_class)
    classifier = DecisionTreeClassifier().fit(X, y_single_class)
    err_msg = 'Got predict_proba of shape \\(10, 1\\), but need classifier with two classes'
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values(classifier, X, response_method=response_method)