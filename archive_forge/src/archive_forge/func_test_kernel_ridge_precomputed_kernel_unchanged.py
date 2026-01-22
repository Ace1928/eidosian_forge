import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._testing import assert_array_almost_equal, ignore_warnings
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_kernel_ridge_precomputed_kernel_unchanged():
    K = np.dot(X, X.T)
    K2 = K.copy()
    KernelRidge(kernel='precomputed').fit(K, y)
    assert_array_almost_equal(K, K2)