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
@ignore_warnings
def test_regression_scorer_sample_weight():
    X, y = make_regression(n_samples=101, n_features=20, random_state=0)
    y = _require_positive_y(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    sample_weight = np.ones_like(y_test)
    sample_weight[:11] = 0
    reg = DecisionTreeRegressor(random_state=0)
    reg.fit(X_train, y_train)
    for name in get_scorer_names():
        scorer = get_scorer(name)
        if name not in REGRESSION_SCORERS:
            continue
        try:
            weighted = scorer(reg, X_test, y_test, sample_weight=sample_weight)
            ignored = scorer(reg, X_test[11:], y_test[11:])
            unweighted = scorer(reg, X_test, y_test)
            assert weighted != unweighted, f'scorer {name} behaves identically when called with sample weights: {weighted} vs {unweighted}'
            assert_almost_equal(weighted, ignored, err_msg=f'scorer {name} behaves differently when ignoring samples and setting sample_weight to 0: {weighted} vs {ignored}')
        except TypeError as e:
            assert 'sample_weight' in str(e), f'scorer {name} raises unhelpful exception when called with sample weights: {str(e)}'