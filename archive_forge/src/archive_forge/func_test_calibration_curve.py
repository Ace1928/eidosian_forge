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
def test_calibration_curve():
    """Check calibration_curve function"""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=2)
    assert len(prob_true) == len(prob_pred)
    assert len(prob_true) == 2
    assert_almost_equal(prob_true, [0, 1])
    assert_almost_equal(prob_pred, [0.1, 0.9])
    with pytest.raises(ValueError):
        calibration_curve([1], [-0.1])
    y_true2 = np.array([0, 0, 0, 0, 1, 1])
    y_pred2 = np.array([0.0, 0.1, 0.2, 0.5, 0.9, 1.0])
    prob_true_quantile, prob_pred_quantile = calibration_curve(y_true2, y_pred2, n_bins=2, strategy='quantile')
    assert len(prob_true_quantile) == len(prob_pred_quantile)
    assert len(prob_true_quantile) == 2
    assert_almost_equal(prob_true_quantile, [0, 2 / 3])
    assert_almost_equal(prob_pred_quantile, [0.1, 0.8])
    with pytest.raises(ValueError):
        calibration_curve(y_true2, y_pred2, strategy='percentile')