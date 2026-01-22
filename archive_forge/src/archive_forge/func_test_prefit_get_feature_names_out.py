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
def test_prefit_get_feature_names_out():
    """Check the interaction between prefit and the feature names."""
    clf = RandomForestClassifier(n_estimators=2, random_state=0)
    clf.fit(data, y)
    model = SelectFromModel(clf, prefit=True, max_features=1)
    name = type(model).__name__
    err_msg = f"This {name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
    with pytest.raises(NotFittedError, match=err_msg):
        model.get_feature_names_out()
    model.fit(data, y)
    feature_names = model.get_feature_names_out()
    assert feature_names == ['x3']