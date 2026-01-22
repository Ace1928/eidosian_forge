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
@pytest.mark.parametrize('estimator, response_method', [(DecisionTreeClassifier(max_depth=2, random_state=0), 'predict_proba'), (DecisionTreeClassifier(max_depth=2, random_state=0), 'predict_log_proba'), (LogisticRegression(), 'decision_function')])
def test_get_response_values_multiclass(estimator, response_method):
    """Check that we can call `_get_response_values` with a multiclass estimator.
    It should return the predictions untouched.
    """
    estimator.fit(X, y)
    predictions, pos_label = _get_response_values(estimator, X, response_method=response_method)
    assert pos_label is None
    assert predictions.shape == (X.shape[0], len(estimator.classes_))
    if response_method == 'predict_proba':
        assert np.logical_and(predictions >= 0, predictions <= 1).all()
    elif response_method == 'predict_log_proba':
        assert (predictions <= 0.0).all()