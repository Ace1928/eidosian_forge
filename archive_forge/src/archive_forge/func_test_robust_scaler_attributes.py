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
@pytest.mark.parametrize('with_centering', [True, False])
@pytest.mark.parametrize('with_scaling', [True, False])
@pytest.mark.parametrize('X', [np.random.randn(10, 3), sparse.rand(10, 3, density=0.5)])
def test_robust_scaler_attributes(X, with_centering, with_scaling):
    if with_centering and sparse.issparse(X):
        pytest.skip('RobustScaler cannot center sparse matrix')
    scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling)
    scaler.fit(X)
    if with_centering:
        assert isinstance(scaler.center_, np.ndarray)
    else:
        assert scaler.center_ is None
    if with_scaling:
        assert isinstance(scaler.scale_, np.ndarray)
    else:
        assert scaler.scale_ is None