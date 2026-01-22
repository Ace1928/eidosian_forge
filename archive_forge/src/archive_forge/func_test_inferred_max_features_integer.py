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
@pytest.mark.parametrize('max_features', [0, 2, data.shape[1], None])
def test_inferred_max_features_integer(max_features):
    """Check max_features_ and output shape for integer max_features."""
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    transformer = SelectFromModel(estimator=clf, max_features=max_features, threshold=-np.inf)
    X_trans = transformer.fit_transform(data, y)
    if max_features is not None:
        assert transformer.max_features_ == max_features
        assert X_trans.shape[1] == transformer.max_features_
    else:
        assert not hasattr(transformer, 'max_features_')
        assert X_trans.shape[1] == data.shape[1]