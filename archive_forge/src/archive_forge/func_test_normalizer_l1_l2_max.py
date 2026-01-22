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
@pytest.mark.parametrize('norm', ['l1', 'l2', 'max'])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_normalizer_l1_l2_max(norm, csr_container):
    rng = np.random.RandomState(0)
    X_dense = rng.randn(4, 5)
    X_sparse_unpruned = csr_container(X_dense)
    X_dense[3, :] = 0.0
    indptr_3 = X_sparse_unpruned.indptr[3]
    indptr_4 = X_sparse_unpruned.indptr[4]
    X_sparse_unpruned.data[indptr_3:indptr_4] = 0.0
    X_sparse_pruned = csr_container(X_dense)
    for X in (X_dense, X_sparse_pruned, X_sparse_unpruned):
        normalizer = Normalizer(norm=norm, copy=True)
        X_norm1 = normalizer.transform(X)
        assert X_norm1 is not X
        X_norm1 = toarray(X_norm1)
        normalizer = Normalizer(norm=norm, copy=False)
        X_norm2 = normalizer.transform(X)
        assert X_norm2 is X
        X_norm2 = toarray(X_norm2)
        for X_norm in (X_norm1, X_norm2):
            check_normalizer(norm, X_norm)