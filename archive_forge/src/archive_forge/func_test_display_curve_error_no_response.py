import numpy as np
import pytest
from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
@pytest.mark.parametrize('response_method, msg', [('predict_proba', 'MyClassifier has none of the following attributes: predict_proba.'), ('decision_function', 'MyClassifier has none of the following attributes: decision_function.'), ('auto', 'MyClassifier has none of the following attributes: predict_proba, decision_function.'), ('bad_method', 'MyClassifier has none of the following attributes: bad_method.')])
@pytest.mark.parametrize('Display', [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay])
def test_display_curve_error_no_response(pyplot, data_binary, response_method, msg, Display):
    """Check that a proper error is raised when the response method requested
    is not defined for the given trained classifier."""
    X, y = data_binary

    class MyClassifier(ClassifierMixin):

        def fit(self, X, y):
            self.classes_ = [0, 1]
            return self
    clf = MyClassifier().fit(X, y)
    with pytest.raises(AttributeError, match=msg):
        Display.from_estimator(clf, X, y, response_method=response_method)