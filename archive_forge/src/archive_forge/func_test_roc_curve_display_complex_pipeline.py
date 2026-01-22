import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import trapezoid
@pytest.mark.parametrize('clf', [LogisticRegression(), make_pipeline(StandardScaler(), LogisticRegression()), make_pipeline(make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression())])
@pytest.mark.parametrize('constructor_name', ['from_estimator', 'from_predictions'])
def test_roc_curve_display_complex_pipeline(pyplot, data_binary, clf, constructor_name):
    """Check the behaviour with complex pipeline."""
    X, y = data_binary
    if constructor_name == 'from_estimator':
        with pytest.raises(NotFittedError):
            RocCurveDisplay.from_estimator(clf, X, y)
    clf.fit(X, y)
    if constructor_name == 'from_estimator':
        display = RocCurveDisplay.from_estimator(clf, X, y)
        name = clf.__class__.__name__
    else:
        display = RocCurveDisplay.from_predictions(y, y)
        name = 'Classifier'
    assert name in display.line_.get_label()
    assert display.estimator_name == name