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
@pytest.mark.parametrize('sparse_container', CSR_CONTAINERS + CSC_CONTAINERS + COO_CONTAINERS + DIA_CONTAINERS)
@pytest.mark.parametrize('output_format', ['csr', 'csc', 'coo'])
def test_check_array_dia_to_int32_indexed_csr_csc_coo(sparse_container, output_format):
    """Check the consistency of the indices dtype with sparse matrices/arrays."""
    X = sparse_container([[0, 1], [1, 0]], dtype=np.float64)
    if hasattr(X, 'offsets'):
        X.offsets = X.offsets.astype(np.int32)
    elif hasattr(X, 'row') and hasattr(X, 'col'):
        X.row = X.row.astype(np.int32)
    elif hasattr(X, 'indices') and hasattr(X, 'indptr'):
        X.indices = X.indices.astype(np.int32)
        X.indptr = X.indptr.astype(np.int32)
    X_checked = check_array(X, accept_sparse=output_format)
    if output_format == 'coo':
        assert X_checked.row.dtype == np.int32
        assert X_checked.col.dtype == np.int32
    else:
        assert X_checked.indices.dtype == np.int32
        assert X_checked.indptr.dtype == np.int32