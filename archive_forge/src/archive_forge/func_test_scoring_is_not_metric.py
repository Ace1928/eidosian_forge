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
def test_scoring_is_not_metric():
    with pytest.raises(ValueError, match='make_scorer'):
        check_scoring(LogisticRegression(), scoring=f1_score)
    with pytest.raises(ValueError, match='make_scorer'):
        check_scoring(LogisticRegression(), scoring=roc_auc_score)
    with pytest.raises(ValueError, match='make_scorer'):
        check_scoring(Ridge(), scoring=r2_score)
    with pytest.raises(ValueError, match='make_scorer'):
        check_scoring(KMeans(), scoring=cluster_module.adjusted_rand_score)
    with pytest.raises(ValueError, match='make_scorer'):
        check_scoring(KMeans(), scoring=cluster_module.rand_score)