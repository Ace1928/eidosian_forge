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
@pytest.mark.parametrize('extension_dtype, regular_dtype', [('boolean', 'bool'), ('Int64', 'int64'), ('Float64', 'float64'), ('category', 'object')])
@pytest.mark.parametrize('include_object', [True, False])
def test_check_array_multiple_extensions(extension_dtype, regular_dtype, include_object):
    """Check pandas extension arrays give the same result as non-extension arrays."""
    pd = pytest.importorskip('pandas')
    X_regular = pd.DataFrame({'a': pd.Series([1, 0, 1, 0], dtype=regular_dtype), 'c': pd.Series([9, 8, 7, 6], dtype='int64')})
    if include_object:
        X_regular['b'] = pd.Series(['a', 'b', 'c', 'd'], dtype='object')
    X_extension = X_regular.assign(a=X_regular['a'].astype(extension_dtype))
    X_regular_checked = check_array(X_regular, dtype=None)
    X_extension_checked = check_array(X_extension, dtype=None)
    assert_array_equal(X_regular_checked, X_extension_checked)