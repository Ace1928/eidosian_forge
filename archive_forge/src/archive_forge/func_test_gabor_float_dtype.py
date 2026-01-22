import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.filters._gabor import _sigma_prefactor, gabor, gabor_kernel
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_gabor_float_dtype(dtype):
    image = np.ones((16, 16), dtype=dtype)
    y = gabor(image, 0.3)
    assert all((arr.dtype == _supported_float_type(image.dtype) for arr in y))