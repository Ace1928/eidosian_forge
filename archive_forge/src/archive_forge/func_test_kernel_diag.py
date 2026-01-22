from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
@pytest.mark.parametrize('kernel', kernels)
def test_kernel_diag(kernel):
    K_call_diag = np.diag(kernel(X))
    K_diag = kernel.diag(X)
    assert_almost_equal(K_call_diag, K_diag, 5)