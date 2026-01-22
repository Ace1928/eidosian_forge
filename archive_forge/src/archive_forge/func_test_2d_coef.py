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
@skip_if_32bit
def test_2d_coef():
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=0, n_repeated=0, shuffle=False, random_state=0, n_classes=4)
    est = LogisticRegression()
    for threshold, func in zip(['mean', 'median'], [np.mean, np.median]):
        for order in [1, 2, np.inf]:
            transformer = SelectFromModel(estimator=LogisticRegression(), threshold=threshold, norm_order=order)
            transformer.fit(X, y)
            assert hasattr(transformer.estimator_, 'coef_')
            X_new = transformer.transform(X)
            assert X_new.shape[1] < X.shape[1]
            est.fit(X, y)
            importances = np.linalg.norm(est.coef_, axis=0, ord=order)
            feature_mask = importances > func(importances)
            assert_array_almost_equal(X_new, X[:, feature_mask])