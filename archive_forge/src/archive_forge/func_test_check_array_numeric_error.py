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
@pytest.mark.parametrize('X', [[['1', '2'], ['3', '4']], np.array([['1', '2'], ['3', '4']], dtype='U'), np.array([['1', '2'], ['3', '4']], dtype='S'), [[b'1', b'2'], [b'3', b'4']], np.array([[b'1', b'2'], [b'3', b'4']], dtype='V1')])
def test_check_array_numeric_error(X):
    """Test that check_array errors when it receives an array of bytes/string
    while a numeric dtype is required."""
    expected_msg = "dtype='numeric' is not compatible with arrays of bytes/strings"
    with pytest.raises(ValueError, match=expected_msg):
        check_array(X, dtype='numeric')