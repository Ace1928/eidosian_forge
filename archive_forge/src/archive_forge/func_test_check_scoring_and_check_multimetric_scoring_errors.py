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
@pytest.mark.parametrize('scoring, msg', [((make_scorer(precision_score), make_scorer(accuracy_score)), 'One or more of the elements were callables'), ([5], 'Non-string types were found'), ((make_scorer(precision_score),), 'One or more of the elements were callables'), ((), 'Empty list was given'), (('f1', 'f1'), 'Duplicate elements were found'), ({4: 'accuracy'}, 'Non-string types were found in the keys'), ({}, 'An empty dict was passed')], ids=['tuple of callables', 'list of int', 'tuple of one callable', 'empty tuple', 'non-unique str', 'non-string key dict', 'empty dict'])
def test_check_scoring_and_check_multimetric_scoring_errors(scoring, msg):
    estimator = EstimatorWithFitAndPredict()
    estimator.fit([[1]], [1])
    with pytest.raises(ValueError, match=msg):
        _check_multimetric_scoring(estimator, scoring=scoring)