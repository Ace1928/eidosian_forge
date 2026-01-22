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
def test_threshold_and_max_features():
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=0, n_repeated=0, shuffle=False, random_state=0)
    est = RandomForestClassifier(n_estimators=50, random_state=0)
    transformer1 = SelectFromModel(estimator=est, max_features=3, threshold=-np.inf)
    X_new1 = transformer1.fit_transform(X, y)
    transformer2 = SelectFromModel(estimator=est, threshold=0.04)
    X_new2 = transformer2.fit_transform(X, y)
    transformer3 = SelectFromModel(estimator=est, max_features=3, threshold=0.04)
    X_new3 = transformer3.fit_transform(X, y)
    assert X_new3.shape[1] == min(X_new1.shape[1], X_new2.shape[1])
    selected_indices = transformer3.transform(np.arange(X.shape[1])[np.newaxis, :])
    assert_allclose(X_new3, X[:, selected_indices[0]])