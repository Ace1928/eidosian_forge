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
@pytest.mark.parametrize('response_method, msg', [('predict_proba', 'MyClassifier has none of the following attributes: predict_proba'), ('decision_function', 'MyClassifier has none of the following attributes: decision_function'), ('auto', 'MyClassifier has none of the following attributes: decision_function, predict_proba, predict'), ('bad_method', 'MyClassifier has none of the following attributes: bad_method')])
def test_error_bad_response(pyplot, response_method, msg):
    """Check errors for bad response."""

    class MyClassifier(BaseEstimator, ClassifierMixin):

        def fit(self, X, y):
            self.fitted_ = True
            self.classes_ = [0, 1]
            return self
    clf = MyClassifier().fit(X, y)
    with pytest.raises(AttributeError, match=msg):
        DecisionBoundaryDisplay.from_estimator(clf, X, response_method=response_method)