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
def test_ordering():
    X = np.ones((10, 5))
    for A in (X, X.T):
        for copy in (True, False):
            B = check_array(A, order='C', copy=copy)
            assert B.flags['C_CONTIGUOUS']
            B = check_array(A, order='F', copy=copy)
            assert B.flags['F_CONTIGUOUS']
            if copy:
                assert A is not B
    X = sp.csr_matrix(X)
    X.data = X.data[::-1]
    assert not X.data.flags['C_CONTIGUOUS']