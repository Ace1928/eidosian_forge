from collections import Counter
import numpy as np
import pytest
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import trapezoid
def test_precision_recall_display_string_labels(pyplot):
    cancer = load_breast_cancer()
    X, y = (cancer.data, cancer.target_names[cancer.target])
    lr = make_pipeline(StandardScaler(), LogisticRegression())
    lr.fit(X, y)
    for klass in cancer.target_names:
        assert klass in lr.classes_
    display = PrecisionRecallDisplay.from_estimator(lr, X, y)
    y_pred = lr.predict_proba(X)[:, 1]
    avg_prec = average_precision_score(y, y_pred, pos_label=lr.classes_[1])
    assert display.average_precision == pytest.approx(avg_prec)
    assert display.estimator_name == lr.__class__.__name__
    err_msg = "y_true takes value in {'benign', 'malignant'}"
    with pytest.raises(ValueError, match=err_msg):
        PrecisionRecallDisplay.from_predictions(y, y_pred)
    display = PrecisionRecallDisplay.from_predictions(y, y_pred, pos_label=lr.classes_[1])
    assert display.average_precision == pytest.approx(avg_prec)