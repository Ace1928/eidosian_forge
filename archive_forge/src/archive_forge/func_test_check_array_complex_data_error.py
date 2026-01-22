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
def test_check_array_complex_data_error():
    X = np.array([[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]])
    with pytest.raises(ValueError, match='Complex data not supported'):
        check_array(X)
    X = [[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]]
    with pytest.raises(ValueError, match='Complex data not supported'):
        check_array(X)
    X = ((1 + 2j, 3 + 4j, 5 + 7j), (2 + 3j, 4 + 5j, 6 + 7j))
    with pytest.raises(ValueError, match='Complex data not supported'):
        check_array(X)
    X = [np.array([1 + 2j, 3 + 4j, 5 + 7j]), np.array([2 + 3j, 4 + 5j, 6 + 7j])]
    with pytest.raises(ValueError, match='Complex data not supported'):
        check_array(X)
    X = (np.array([1 + 2j, 3 + 4j, 5 + 7j]), np.array([2 + 3j, 4 + 5j, 6 + 7j]))
    with pytest.raises(ValueError, match='Complex data not supported'):
        check_array(X)
    X = MockDataFrame(np.array([[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]]))
    with pytest.raises(ValueError, match='Complex data not supported'):
        check_array(X)
    X = sp.coo_matrix([[0, 1 + 2j], [0, 0]])
    with pytest.raises(ValueError, match='Complex data not supported'):
        check_array(X)
    y = np.array([1 + 2j, 3 + 4j, 5 + 7j, 2 + 3j, 4 + 5j, 6 + 7j])
    with pytest.raises(ValueError, match='Complex data not supported'):
        _check_y(y)