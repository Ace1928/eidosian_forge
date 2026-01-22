from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
@pytest.mark.parametrize('kernel', [kernel for kernel in kernels if kernel.is_stationary()])
def test_kernel_stationary(kernel):
    K = kernel(X, X + 1)
    assert_almost_equal(K[0, 0], np.diag(K))