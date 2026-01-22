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
def test_check_array_series_err_msg():
    """
    Check that we raise a proper error message when passing a Series and we expect a
    2-dimensional container.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27498
    """
    pd = pytest.importorskip('pandas')
    ser = pd.Series([1, 2, 3])
    msg = f'Expected a 2-dimensional container but got {type(ser)} instead.'
    with pytest.raises(ValueError, match=msg):
        check_array(ser, ensure_2d=True)