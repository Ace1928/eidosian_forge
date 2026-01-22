import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.filters._gabor import _sigma_prefactor, gabor, gabor_kernel
def test_gabor_kernel_size():
    sigma_x = 5
    sigma_y = 10
    size_x = sigma_x * 6 + 1
    size_y = sigma_y * 6 + 1
    kernel = gabor_kernel(0, theta=0, sigma_x=sigma_x, sigma_y=sigma_y)
    assert_equal(kernel.shape, (size_y, size_x))
    kernel = gabor_kernel(0, theta=np.pi / 2, sigma_x=sigma_x, sigma_y=sigma_y)
    assert_equal(kernel.shape, (size_x, size_y))