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
def test_scorer_no_op_multiclass_select_proba():
    X, y = make_classification(n_classes=3, n_informative=3, n_samples=20, random_state=0)
    lr = LogisticRegression().fit(X, y)
    mask_last_class = y == lr.classes_[-1]
    X_test, y_test = (X[~mask_last_class], y[~mask_last_class])
    assert_array_equal(np.unique(y_test), lr.classes_[:-1])
    scorer = make_scorer(roc_auc_score, response_method='predict_proba', multi_class='ovo', labels=lr.classes_)
    scorer(lr, X_test, y_test)