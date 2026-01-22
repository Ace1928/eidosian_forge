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
@pytest.mark.parametrize('scorer, expected_repr', [(get_scorer('accuracy'), "make_scorer(accuracy_score, response_method='predict')"), (get_scorer('neg_log_loss'), "make_scorer(log_loss, greater_is_better=False, response_method='predict_proba')"), (get_scorer('roc_auc'), "make_scorer(roc_auc_score, response_method=('decision_function', 'predict_proba'))"), (make_scorer(fbeta_score, beta=2), "make_scorer(fbeta_score, response_method='predict', beta=2)")])
def test_make_scorer_repr(scorer, expected_repr):
    """Check the representation of the scorer."""
    assert repr(scorer) == expected_repr