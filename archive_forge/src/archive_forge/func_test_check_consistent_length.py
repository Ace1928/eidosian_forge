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
def test_check_consistent_length():
    check_consistent_length([1], [2], [3], [4], [5])
    check_consistent_length([[1, 2], [[1, 2]]], [1, 2], ['a', 'b'])
    check_consistent_length([1], (2,), np.array([3]), sp.csr_matrix((1, 2)))
    with pytest.raises(ValueError, match='inconsistent numbers of samples'):
        check_consistent_length([1, 2], [1])
    with pytest.raises(TypeError, match="got <\\w+ 'int'>"):
        check_consistent_length([1, 2], 1)
    with pytest.raises(TypeError, match="got <\\w+ 'object'>"):
        check_consistent_length([1, 2], object())
    with pytest.raises(TypeError):
        check_consistent_length([1, 2], np.array(1))
    with pytest.raises(TypeError, match='Expected sequence or array-like'):
        check_consistent_length([1, 2], RandomForestRegressor())