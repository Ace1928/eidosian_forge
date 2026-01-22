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
@ignore_warnings
def test_check_array():
    X = [[1, 2], [3, 4]]
    X_csr = sp.csr_matrix(X)
    with pytest.raises(TypeError):
        check_array(X_csr)
    X_array = check_array([0, 1, 2], ensure_2d=False)
    assert X_array.ndim == 1
    with pytest.raises(ValueError, match='Expected 2D array, got 1D array instead'):
        check_array([0, 1, 2], ensure_2d=True)
    with pytest.raises(ValueError, match='Expected 2D array, got scalar array instead'):
        check_array(10, ensure_2d=True)
    X_ndim = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError):
        check_array(X_ndim)
    check_array(X_ndim, allow_nd=True)
    X_C = np.arange(4).reshape(2, 2).copy('C')
    X_F = X_C.copy('F')
    X_int = X_C.astype(int)
    X_float = X_C.astype(float)
    Xs = [X_C, X_F, X_int, X_float]
    dtypes = [np.int32, int, float, np.float32, None, bool, object]
    orders = ['C', 'F', None]
    copys = [True, False]
    for X, dtype, order, copy in product(Xs, dtypes, orders, copys):
        X_checked = check_array(X, dtype=dtype, order=order, copy=copy)
        if dtype is not None:
            assert X_checked.dtype == dtype
        else:
            assert X_checked.dtype == X.dtype
        if order == 'C':
            assert X_checked.flags['C_CONTIGUOUS']
            assert not X_checked.flags['F_CONTIGUOUS']
        elif order == 'F':
            assert X_checked.flags['F_CONTIGUOUS']
            assert not X_checked.flags['C_CONTIGUOUS']
        if copy:
            assert X is not X_checked
        elif X.dtype == X_checked.dtype and X_checked.flags['C_CONTIGUOUS'] == X.flags['C_CONTIGUOUS'] and (X_checked.flags['F_CONTIGUOUS'] == X.flags['F_CONTIGUOUS']):
            assert X is X_checked
    Xs = []
    Xs.extend([sparse_container(X_C) for sparse_container in CSR_CONTAINERS + CSC_CONTAINERS + COO_CONTAINERS + DOK_CONTAINERS])
    Xs.extend([Xs[0].astype(np.int64), Xs[0].astype(np.float64)])
    accept_sparses = [['csr', 'coo'], ['coo', 'dok']]
    non_object_dtypes = [dt for dt in dtypes if dt is not object]
    for X, dtype, accept_sparse, copy in product(Xs, non_object_dtypes, accept_sparses, copys):
        X_checked = check_array(X, dtype=dtype, accept_sparse=accept_sparse, copy=copy)
        if dtype is not None:
            assert X_checked.dtype == dtype
        else:
            assert X_checked.dtype == X.dtype
        if X.format in accept_sparse:
            assert X.format == X_checked.format
        else:
            assert X_checked.format == accept_sparse[0]
        if copy:
            assert X is not X_checked
        elif X.dtype == X_checked.dtype and X.format == X_checked.format:
            assert X is X_checked
    X_dense = check_array([[1, 2], [3, 4]])
    assert isinstance(X_dense, np.ndarray)
    with pytest.raises(ValueError):
        check_array(X_ndim.tolist())
    check_array(X_ndim.tolist(), allow_nd=True)
    X_no_array = _NotAnArray(X_dense)
    result = check_array(X_no_array)
    assert isinstance(result, np.ndarray)