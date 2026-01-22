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
def test_average_precision_pos_label(string_labeled_classification_problem):
    clf, X_test, y_test, _, y_pred_proba, y_pred_decision = string_labeled_classification_problem
    pos_label = 'cancer'
    y_pred_proba = y_pred_proba[:, 0]
    y_pred_decision = y_pred_decision * -1
    assert clf.classes_[0] == pos_label
    ap_proba = average_precision_score(y_test, y_pred_proba, pos_label=pos_label)
    ap_decision_function = average_precision_score(y_test, y_pred_decision, pos_label=pos_label)
    assert ap_proba == pytest.approx(ap_decision_function)
    average_precision_scorer = make_scorer(average_precision_score, response_method=('decision_function', 'predict_proba'))
    err_msg = 'pos_label=1 is not a valid label. It should be one of '
    with pytest.raises(ValueError, match=err_msg):
        average_precision_scorer(clf, X_test, y_test)
    average_precision_scorer = make_scorer(average_precision_score, response_method=('decision_function', 'predict_proba'), pos_label=pos_label)
    ap_scorer = average_precision_scorer(clf, X_test, y_test)
    assert ap_scorer == pytest.approx(ap_proba)
    clf_without_predict_proba = deepcopy(clf)

    def _predict_proba(self, X):
        raise NotImplementedError
    clf_without_predict_proba.predict_proba = partial(_predict_proba, clf_without_predict_proba)
    with pytest.raises(NotImplementedError):
        clf_without_predict_proba.predict_proba(X_test)
    ap_scorer = average_precision_scorer(clf_without_predict_proba, X_test, y_test)
    assert ap_scorer == pytest.approx(ap_proba)