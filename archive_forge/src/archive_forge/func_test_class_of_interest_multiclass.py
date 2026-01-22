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
@pytest.mark.parametrize('response_method', ['predict_proba', 'decision_function'])
def test_class_of_interest_multiclass(pyplot, response_method):
    """Check the behaviour of passing `class_of_interest` for plotting the output of
    `predict_proba` and `decision_function` in the multiclass case.
    """
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    class_of_interest_idx = 2
    estimator = LogisticRegression().fit(X, y)
    disp = DecisionBoundaryDisplay.from_estimator(estimator, X, response_method=response_method, class_of_interest=class_of_interest_idx)
    grid = np.concatenate([disp.xx0.reshape(-1, 1), disp.xx1.reshape(-1, 1)], axis=1)
    response = getattr(estimator, response_method)(grid)[:, class_of_interest_idx]
    assert_allclose(response.reshape(*disp.response.shape), disp.response)
    y = iris.target_names[iris.target]
    estimator = LogisticRegression().fit(X, y)
    disp = DecisionBoundaryDisplay.from_estimator(estimator, X, response_method=response_method, class_of_interest=iris.target_names[class_of_interest_idx])
    grid = np.concatenate([disp.xx0.reshape(-1, 1), disp.xx1.reshape(-1, 1)], axis=1)
    response = getattr(estimator, response_method)(grid)[:, class_of_interest_idx]
    assert_allclose(response.reshape(*disp.response.shape), disp.response)
    err_msg = 'class_of_interest=2 is not a valid label: It should be one of'
    with pytest.raises(ValueError, match=err_msg):
        DecisionBoundaryDisplay.from_estimator(estimator, X, response_method=response_method, class_of_interest=class_of_interest_idx)
    err_msg = 'Multiclass classifiers are only supported'
    with pytest.raises(ValueError, match=err_msg):
        DecisionBoundaryDisplay.from_estimator(estimator, X, response_method=response_method, class_of_interest=None)