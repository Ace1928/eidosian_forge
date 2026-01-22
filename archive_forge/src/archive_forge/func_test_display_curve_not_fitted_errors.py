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
@pytest.mark.parametrize('clf', [LogisticRegression(), make_pipeline(StandardScaler(), LogisticRegression()), make_pipeline(make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression())])
@pytest.mark.parametrize('Display', [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay])
def test_display_curve_not_fitted_errors(pyplot, data_binary, clf, Display):
    """Check that a proper error is raised when the classifier is not
    fitted."""
    X, y = data_binary
    model = clone(clf)
    with pytest.raises(NotFittedError):
        Display.from_estimator(model, X, y)
    model.fit(X, y)
    disp = Display.from_estimator(model, X, y)
    assert model.__class__.__name__ in disp.line_.get_label()
    assert disp.estimator_name == model.__class__.__name__