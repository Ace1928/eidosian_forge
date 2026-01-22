import re
import warnings
from unittest.mock import Mock
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import make_friedman1
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.utils._testing import (
@pytest.mark.parametrize('error, err_msg, max_features', ([ValueError, 'max_features == 10, must be <= 4', 10], [ValueError, 'max_features == 5, must be <= 4', lambda x: x.shape[1] + 1]))
def test_partial_fit_validate_max_features(error, err_msg, max_features):
    """Test that partial_fit from SelectFromModel validates `max_features`."""
    X, y = datasets.make_classification(n_samples=100, n_features=4, random_state=0)
    with pytest.raises(error, match=err_msg):
        SelectFromModel(estimator=SGDClassifier(), max_features=max_features).partial_fit(X, y, classes=[0, 1])