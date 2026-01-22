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
def test_check_is_fitted_attributes():

    class MyEstimator:

        def fit(self, X, y):
            return self
    msg = 'not fitted'
    est = MyEstimator()
    assert not _is_fitted(est, attributes=['a_', 'b_'])
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=['a_', 'b_'])
    assert not _is_fitted(est, attributes=['a_', 'b_'], all_or_any=all)
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=['a_', 'b_'], all_or_any=all)
    assert not _is_fitted(est, attributes=['a_', 'b_'], all_or_any=any)
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=['a_', 'b_'], all_or_any=any)
    est.a_ = 'a'
    assert not _is_fitted(est, attributes=['a_', 'b_'])
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=['a_', 'b_'])
    assert not _is_fitted(est, attributes=['a_', 'b_'], all_or_any=all)
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=['a_', 'b_'], all_or_any=all)
    assert _is_fitted(est, attributes=['a_', 'b_'], all_or_any=any)
    check_is_fitted(est, attributes=['a_', 'b_'], all_or_any=any)
    est.b_ = 'b'
    assert _is_fitted(est, attributes=['a_', 'b_'])
    check_is_fitted(est, attributes=['a_', 'b_'])
    assert _is_fitted(est, attributes=['a_', 'b_'], all_or_any=all)
    check_is_fitted(est, attributes=['a_', 'b_'], all_or_any=all)
    assert _is_fitted(est, attributes=['a_', 'b_'], all_or_any=any)
    check_is_fitted(est, attributes=['a_', 'b_'], all_or_any=any)