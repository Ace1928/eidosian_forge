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
def test_max_features_tiebreak():
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=0, n_repeated=0, shuffle=False, random_state=0)
    max_features = X.shape[1]
    feature_importances = np.array([4, 4, 4, 4, 3, 3, 3, 2, 2, 1])
    for n_features in range(1, max_features + 1):
        transformer = SelectFromModel(FixedImportanceEstimator(feature_importances), max_features=n_features, threshold=-np.inf)
        X_new = transformer.fit_transform(X, y)
        selected_feature_indices = np.where(transformer._get_support_mask())[0]
        assert_array_equal(selected_feature_indices, np.arange(n_features))
        assert X_new.shape[1] == n_features