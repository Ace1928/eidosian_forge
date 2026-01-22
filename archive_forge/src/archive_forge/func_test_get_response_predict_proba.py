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
def test_get_response_predict_proba():
    """Check the behaviour of `_get_response_values_binary` using `predict_proba`."""
    classifier = DecisionTreeClassifier().fit(X_binary, y_binary)
    y_proba, pos_label = _get_response_values_binary(classifier, X_binary, response_method='predict_proba')
    assert_allclose(y_proba, classifier.predict_proba(X_binary)[:, 1])
    assert pos_label == 1
    y_proba, pos_label = _get_response_values_binary(classifier, X_binary, response_method='predict_proba', pos_label=0)
    assert_allclose(y_proba, classifier.predict_proba(X_binary)[:, 0])
    assert pos_label == 0