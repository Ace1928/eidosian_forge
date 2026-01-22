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
@pytest.mark.parametrize('constructor_name', ['from_estimator', 'from_predictions'])
def test_calibration_display_name_multiple_calls(constructor_name, pyplot, iris_data_binary):
    X, y = iris_data_binary
    clf_name = 'my hand-crafted name'
    clf = LogisticRegression().fit(X, y)
    y_prob = clf.predict_proba(X)[:, 1]
    constructor = getattr(CalibrationDisplay, constructor_name)
    params = (clf, X, y) if constructor_name == 'from_estimator' else (y, y_prob)
    viz = constructor(*params, name=clf_name)
    assert viz.estimator_name == clf_name
    pyplot.close('all')
    viz.plot()
    expected_legend_labels = [clf_name, 'Perfectly calibrated']
    legend_labels = viz.ax_.get_legend().get_texts()
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels
    pyplot.close('all')
    clf_name = 'another_name'
    viz.plot(name=clf_name)
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels