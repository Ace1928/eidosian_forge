import numpy as np
import pytest
from numpy.testing import (
from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
def test_confusion_matrix_display_validation(pyplot):
    """Check that we raise the proper error when validating parameters."""
    X, y = make_classification(n_samples=100, n_informative=5, n_classes=5, random_state=0)
    with pytest.raises(NotFittedError):
        ConfusionMatrixDisplay.from_estimator(SVC(), X, y)
    regressor = SVR().fit(X, y)
    y_pred_regressor = regressor.predict(X)
    y_pred_classifier = SVC().fit(X, y).predict(X)
    err_msg = 'ConfusionMatrixDisplay.from_estimator only supports classifiers'
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_estimator(regressor, X, y)
    err_msg = 'Mix type of y not allowed, got types'
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_predictions(y + 0.5, y_pred_classifier)
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_predictions(y, y_pred_regressor)
    err_msg = 'Found input variables with inconsistent numbers of samples'
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_predictions(y, y_pred_classifier[::2])