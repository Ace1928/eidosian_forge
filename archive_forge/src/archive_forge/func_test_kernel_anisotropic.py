from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
def test_kernel_anisotropic():
    kernel = 3.0 * RBF([0.5, 2.0])
    K = kernel(X)
    X1 = np.array(X)
    X1[:, 0] *= 4
    K1 = 3.0 * RBF(2.0)(X1)
    assert_almost_equal(K, K1)
    X2 = np.array(X)
    X2[:, 1] /= 4
    K2 = 3.0 * RBF(0.5)(X2)
    assert_almost_equal(K, K2)
    kernel.theta = kernel.theta + np.log(2)
    assert_array_equal(kernel.theta, np.log([6.0, 1.0, 4.0]))
    assert_array_equal(kernel.k2.length_scale, [1.0, 4.0])