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
@pytest.mark.parametrize('n_bins', [5, 10])
@pytest.mark.parametrize('strategy', ['uniform', 'quantile'])
def test_calibration_display_compute(pyplot, iris_data_binary, n_bins, strategy):
    X, y = iris_data_binary
    lr = LogisticRegression().fit(X, y)
    viz = CalibrationDisplay.from_estimator(lr, X, y, n_bins=n_bins, strategy=strategy, alpha=0.8)
    y_prob = lr.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=n_bins, strategy=strategy)
    assert_allclose(viz.prob_true, prob_true)
    assert_allclose(viz.prob_pred, prob_pred)
    assert_allclose(viz.y_prob, y_prob)
    assert viz.estimator_name == 'LogisticRegression'
    import matplotlib as mpl
    assert isinstance(viz.line_, mpl.lines.Line2D)
    assert viz.line_.get_alpha() == 0.8
    assert isinstance(viz.ax_, mpl.axes.Axes)
    assert isinstance(viz.figure_, mpl.figure.Figure)
    assert viz.ax_.get_xlabel() == 'Mean predicted probability (Positive class: 1)'
    assert viz.ax_.get_ylabel() == 'Fraction of positives (Positive class: 1)'
    expected_legend_labels = ['LogisticRegression', 'Perfectly calibrated']
    legend_labels = viz.ax_.get_legend().get_texts()
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels