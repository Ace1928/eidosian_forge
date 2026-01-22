import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS)
def test_scaler_return_identity(sparse_container):
    X_dense = np.array([[0, 1, 3], [5, 6, 0], [8, 0, 10]], dtype=np.float64)
    X_sparse = sparse_container(X_dense)
    transformer_dense = StandardScaler(with_mean=False, with_std=False)
    X_trans_dense = transformer_dense.fit_transform(X_dense)
    assert_allclose(X_trans_dense, X_dense)
    transformer_sparse = clone(transformer_dense)
    X_trans_sparse = transformer_sparse.fit_transform(X_sparse)
    assert_allclose_dense_sparse(X_trans_sparse, X_sparse)
    _check_identity_scalers_attributes(transformer_dense, transformer_sparse)
    transformer_dense.partial_fit(X_dense)
    transformer_sparse.partial_fit(X_sparse)
    _check_identity_scalers_attributes(transformer_dense, transformer_sparse)
    transformer_dense.fit(X_dense)
    transformer_sparse.fit(X_sparse)
    _check_identity_scalers_attributes(transformer_dense, transformer_sparse)