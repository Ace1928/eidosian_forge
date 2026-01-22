import numbers
import os
import pickle
import shutil
import tempfile
from copy import deepcopy
from functools import partial
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.linear_model import LogisticRegression, Perceptron, Ridge
from sklearn.metrics import (
from sklearn.metrics import cluster as cluster_module
from sklearn.metrics._scorer import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
from sklearn.utils.metadata_routing import MetadataRouter
@pytest.mark.parametrize('deprecated_params, new_params, warn_msg', [({'needs_proba': True}, {'response_method': 'predict_proba'}, 'The `needs_threshold` and `needs_proba` parameter are deprecated'), ({'needs_proba': True, 'needs_threshold': False}, {'response_method': 'predict_proba'}, 'The `needs_threshold` and `needs_proba` parameter are deprecated'), ({'needs_threshold': True}, {'response_method': ('decision_function', 'predict_proba')}, 'The `needs_threshold` and `needs_proba` parameter are deprecated'), ({'needs_threshold': True, 'needs_proba': False}, {'response_method': ('decision_function', 'predict_proba')}, 'The `needs_threshold` and `needs_proba` parameter are deprecated'), ({'needs_threshold': False, 'needs_proba': False}, {'response_method': 'predict'}, 'The `needs_threshold` and `needs_proba` parameter are deprecated')])
def test_make_scorer_deprecation(deprecated_params, new_params, warn_msg):
    """Check that we raise a deprecation warning when using `needs_proba` or
    `needs_threshold`."""
    X, y = make_classification(n_samples=150, n_features=10, random_state=0)
    classifier = LogisticRegression().fit(X, y)
    with pytest.warns(FutureWarning, match=warn_msg):
        deprecated_roc_auc_scorer = make_scorer(roc_auc_score, **deprecated_params)
    roc_auc_scorer = make_scorer(roc_auc_score, **new_params)
    assert deprecated_roc_auc_scorer(classifier, X, y) == pytest.approx(roc_auc_scorer(classifier, X, y))