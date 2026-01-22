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
@pytest.mark.parametrize('lambdas, expected_lambdas, w_type, w_msg', list(_psd_cases_valid.values()), ids=list(_psd_cases_valid.keys()))
@pytest.mark.parametrize('enable_warnings', [True, False])
def test_check_psd_eigenvalues_valid(lambdas, expected_lambdas, w_type, w_msg, enable_warnings):
    if not enable_warnings:
        w_type = None
    if w_type is None:
        with warnings.catch_warnings():
            warnings.simplefilter('error', PositiveSpectrumWarning)
            lambdas_fixed = _check_psd_eigenvalues(lambdas, enable_warnings=enable_warnings)
    else:
        with pytest.warns(w_type, match=w_msg):
            lambdas_fixed = _check_psd_eigenvalues(lambdas, enable_warnings=enable_warnings)
    assert_allclose(expected_lambdas, lambdas_fixed)