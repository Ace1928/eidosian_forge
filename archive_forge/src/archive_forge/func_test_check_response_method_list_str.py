import numbers
import re
import warnings
from itertools import product
from operator import itemgetter
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
from pytest import importorskip
import sklearn
from sklearn._config import config_context
from sklearn._min_dependencies import dependent_packages
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError, PositiveSpectrumWarning
from sklearn.linear_model import ARDRegression
from sklearn.metrics.tests.test_score_objects import EstimatorWithFit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import _sparse_random_matrix
from sklearn.svm import SVR
from sklearn.utils import (
from sklearn.utils._mocking import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.validation import (
def test_check_response_method_list_str():
    """Check that we can pass a list of ordered method."""
    method_implemented = ['predict_proba']
    my_estimator = _MockEstimatorOnOffPrediction(method_implemented)
    X = 'mocking_data'
    response_method = ['decision_function', 'predict']
    err_msg = f'_MockEstimatorOnOffPrediction has none of the following attributes: {', '.join(response_method)}.'
    with pytest.raises(AttributeError, match=err_msg):
        _check_response_method(my_estimator, response_method)(X)
    response_method = ['decision_function', 'predict_proba']
    method_name_predicting = _check_response_method(my_estimator, response_method)(X)
    assert method_name_predicting == 'predict_proba'
    method_implemented = ['predict_proba', 'predict']
    my_estimator = _MockEstimatorOnOffPrediction(method_implemented)
    response_method = ['decision_function', 'predict', 'predict_proba']
    method_name_predicting = _check_response_method(my_estimator, response_method)(X)
    assert method_name_predicting == 'predict'