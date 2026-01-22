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
@pytest.mark.parametrize('density', [0, 0.05, 0.1, 0.5, 1])
@pytest.mark.parametrize('strictly_signed', ['positive', 'negative', 'zeros', None])
def test_robust_scaler_equivalence_dense_sparse(density, strictly_signed):
    X_sparse = sparse.rand(1000, 5, density=density).tocsc()
    if strictly_signed == 'positive':
        X_sparse.data = np.abs(X_sparse.data)
    elif strictly_signed == 'negative':
        X_sparse.data = -np.abs(X_sparse.data)
    elif strictly_signed == 'zeros':
        X_sparse.data = np.zeros(X_sparse.data.shape, dtype=np.float64)
    X_dense = X_sparse.toarray()
    scaler_sparse = RobustScaler(with_centering=False)
    scaler_dense = RobustScaler(with_centering=False)
    scaler_sparse.fit(X_sparse)
    scaler_dense.fit(X_dense)
    assert_allclose(scaler_sparse.scale_, scaler_dense.scale_)