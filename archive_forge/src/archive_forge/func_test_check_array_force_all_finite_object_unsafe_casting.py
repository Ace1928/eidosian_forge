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
@pytest.mark.parametrize('X, err_msg', [(np.array([[1, np.nan]]), 'Input contains NaN.'), (np.array([[1, np.nan]]), 'Input contains NaN.'), (np.array([[1, np.inf]]), 'Input contains infinity or a value too large for.*int'), (np.array([[1, np.nan]], dtype=object), 'cannot convert float NaN to integer')])
@pytest.mark.parametrize('force_all_finite', [True, False])
def test_check_array_force_all_finite_object_unsafe_casting(X, err_msg, force_all_finite):
    with pytest.raises(ValueError, match=err_msg):
        check_array(X, dtype=int, force_all_finite=force_all_finite)