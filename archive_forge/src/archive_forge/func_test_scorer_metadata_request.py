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
@pytest.mark.parametrize('name', get_scorer_names(), ids=get_scorer_names())
def test_scorer_metadata_request(name):
    """Testing metadata requests for scorers.

    This test checks many small things in a large test, to reduce the
    boilerplate required for each section.
    """
    scorer = get_scorer(name)
    assert hasattr(scorer, 'set_score_request')
    assert hasattr(scorer, 'get_metadata_routing')
    assert_request_is_empty(scorer.get_metadata_routing())
    weighted_scorer = scorer.set_score_request(sample_weight=True)
    assert weighted_scorer is scorer
    assert_request_is_empty(weighted_scorer.get_metadata_routing(), exclude='score')
    assert weighted_scorer.get_metadata_routing().score.requests['sample_weight'] is True
    router = MetadataRouter(owner='test').add(method_mapping='score', scorer=get_scorer(name))
    with pytest.raises(TypeError, match='got unexpected argument'):
        router.validate_metadata(params={'sample_weight': 1}, method='score')
    routed_params = router.route_params(params={'sample_weight': 1}, caller='score')
    assert not routed_params.scorer.score
    router = MetadataRouter(owner='test').add(scorer=weighted_scorer, method_mapping='score')
    router.validate_metadata(params={'sample_weight': 1}, method='score')
    routed_params = router.route_params(params={'sample_weight': 1}, caller='score')
    assert list(routed_params.scorer.score.keys()) == ['sample_weight']