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
@pytest.mark.parametrize('enable_metadata_routing', [True, False])
def test_metadata_routing_multimetric_metadata_routing(enable_metadata_routing):
    """Test multimetric scorer works with and without metadata routing enabled when
    there is no actual metadata to pass.

    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/28256
    """
    X, y = make_classification(n_samples=50, n_features=10, random_state=0)
    estimator = EstimatorWithFitAndPredict().fit(X, y)
    multimetric_scorer = _MultimetricScorer(scorers={'acc': get_scorer('accuracy')})
    with config_context(enable_metadata_routing=enable_metadata_routing):
        multimetric_scorer(estimator, X, y)