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
def test_check_symmetric():
    arr_sym = np.array([[0, 1], [1, 2]])
    arr_bad = np.ones(2)
    arr_asym = np.array([[0, 2], [0, 2]])
    test_arrays = {'dense': arr_asym, 'dok': sp.dok_matrix(arr_asym), 'csr': sp.csr_matrix(arr_asym), 'csc': sp.csc_matrix(arr_asym), 'coo': sp.coo_matrix(arr_asym), 'lil': sp.lil_matrix(arr_asym), 'bsr': sp.bsr_matrix(arr_asym)}
    with pytest.raises(ValueError):
        check_symmetric(arr_bad)
    for arr_format, arr in test_arrays.items():
        with pytest.warns(UserWarning):
            check_symmetric(arr)
        with pytest.raises(ValueError):
            check_symmetric(arr, raise_exception=True)
        output = check_symmetric(arr, raise_warning=False)
        if sp.issparse(output):
            assert output.format == arr_format
            assert_array_equal(output.toarray(), arr_sym)
        else:
            assert_array_equal(output, arr_sym)