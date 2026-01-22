import warnings
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import (
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
@pytest.mark.parametrize('estimator, response_method, class_of_interest, expected_prediction_method', [(DecisionTreeRegressor(), 'predict', None, 'predict'), (DecisionTreeRegressor(), 'auto', None, 'predict'), (LogisticRegression().fit(*load_iris_2d_scaled()), 'predict', None, 'predict'), (LogisticRegression().fit(*load_iris_2d_scaled()), 'auto', None, 'predict'), (LogisticRegression().fit(*load_iris_2d_scaled()), 'predict_proba', 0, 'predict_proba'), (LogisticRegression().fit(*load_iris_2d_scaled()), 'decision_function', 0, 'decision_function'), (LogisticRegression().fit(X, y), 'auto', None, ['decision_function', 'predict_proba', 'predict']), (LogisticRegression().fit(X, y), 'predict', None, 'predict'), (LogisticRegression().fit(X, y), ['predict_proba', 'decision_function'], None, ['predict_proba', 'decision_function'])])
def test_check_boundary_response_method(estimator, response_method, class_of_interest, expected_prediction_method):
    """Check the behaviour of `_check_boundary_response_method` for the supported
    cases.
    """
    prediction_method = _check_boundary_response_method(estimator, response_method, class_of_interest)
    assert prediction_method == expected_prediction_method