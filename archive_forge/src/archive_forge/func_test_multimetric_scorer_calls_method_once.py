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
@pytest.mark.parametrize('scorers,expected_predict_count,expected_predict_proba_count,expected_decision_func_count', [({'a1': 'accuracy', 'a2': 'accuracy', 'll1': 'neg_log_loss', 'll2': 'neg_log_loss', 'ra1': 'roc_auc', 'ra2': 'roc_auc'}, 1, 1, 1), (['roc_auc', 'accuracy'], 1, 0, 1), (['neg_log_loss', 'accuracy'], 1, 1, 0)])
def test_multimetric_scorer_calls_method_once(scorers, expected_predict_count, expected_predict_proba_count, expected_decision_func_count):
    X, y = (np.array([[1], [1], [0], [0], [0]]), np.array([0, 1, 1, 1, 0]))
    mock_est = Mock()
    mock_est._estimator_type = 'classifier'
    fit_func = Mock(return_value=mock_est, name='fit')
    fit_func.__name__ = 'fit'
    predict_func = Mock(return_value=y, name='predict')
    predict_func.__name__ = 'predict'
    pos_proba = np.random.rand(X.shape[0])
    proba = np.c_[1 - pos_proba, pos_proba]
    predict_proba_func = Mock(return_value=proba, name='predict_proba')
    predict_proba_func.__name__ = 'predict_proba'
    decision_function_func = Mock(return_value=pos_proba, name='decision_function')
    decision_function_func.__name__ = 'decision_function'
    mock_est.fit = fit_func
    mock_est.predict = predict_func
    mock_est.predict_proba = predict_proba_func
    mock_est.decision_function = decision_function_func
    mock_est.classes_ = np.array([0, 1])
    scorer_dict = _check_multimetric_scoring(LogisticRegression(), scorers)
    multi_scorer = _MultimetricScorer(scorers=scorer_dict)
    results = multi_scorer(mock_est, X, y)
    assert set(scorers) == set(results)
    assert predict_func.call_count == expected_predict_count
    assert predict_proba_func.call_count == expected_predict_proba_count
    assert decision_function_func.call_count == expected_decision_func_count