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
@pytest.mark.usefixtures('enable_slep006')
def test_multimetric_scoring_metadata_routing():

    def score1(y_true, y_pred):
        return 1

    def score2(y_true, y_pred, sample_weight='test'):
        assert sample_weight == 'test'
        return 1

    def score3(y_true, y_pred, sample_weight=None):
        assert sample_weight is not None
        return 1
    scorers = {'score1': make_scorer(score1), 'score2': make_scorer(score2).set_score_request(sample_weight=False), 'score3': make_scorer(score3).set_score_request(sample_weight=True)}
    X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, random_state=0)
    clf = DecisionTreeClassifier().fit(X, y)
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    multi_scorer = _MultimetricScorer(scorers=scorer_dict)
    with config_context(enable_metadata_routing=False):
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            multi_scorer(clf, X, y, sample_weight=1)
    multi_scorer(clf, X, y, sample_weight=1)