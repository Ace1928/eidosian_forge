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
def test_calibration_display_ref_line(pyplot, iris_data_binary):
    X, y = iris_data_binary
    lr = LogisticRegression().fit(X, y)
    dt = DecisionTreeClassifier().fit(X, y)
    viz = CalibrationDisplay.from_estimator(lr, X, y)
    viz2 = CalibrationDisplay.from_estimator(dt, X, y, ax=viz.ax_)
    labels = viz2.ax_.get_legend_handles_labels()[1]
    assert labels.count('Perfectly calibrated') == 1