import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import (
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('pos_label, expected_pos_label', [(None, 1), (0, 0), (1, 1)])
def test_calibration_display_pos_label(pyplot, iris_data_binary, pos_label, expected_pos_label):
    """Check the behaviour of `pos_label` in the `CalibrationDisplay`."""
    X, y = iris_data_binary
    lr = LogisticRegression().fit(X, y)
    viz = CalibrationDisplay.from_estimator(lr, X, y, pos_label=pos_label)
    y_prob = lr.predict_proba(X)[:, expected_pos_label]
    prob_true, prob_pred = calibration_curve(y, y_prob, pos_label=pos_label)
    assert_allclose(viz.prob_true, prob_true)
    assert_allclose(viz.prob_pred, prob_pred)
    assert_allclose(viz.y_prob, y_prob)
    assert viz.ax_.get_xlabel() == f'Mean predicted probability (Positive class: {expected_pos_label})'
    assert viz.ax_.get_ylabel() == f'Fraction of positives (Positive class: {expected_pos_label})'
    expected_legend_labels = [lr.__class__.__name__, 'Perfectly calibrated']
    legend_labels = viz.ax_.get_legend().get_texts()
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels