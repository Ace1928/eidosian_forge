import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from scipy.ndimage import map_coordinates
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color.colorconv import rgb2gray
from skimage.data import checkerboard, astronaut
from skimage.draw.draw import circle_perimeter_aa
from skimage.feature.peak import peak_local_max
from skimage.transform._warps import (
from skimage.transform._geometric import (
from skimage.util.dtype import img_as_float, _convert
@pytest.mark.parametrize('dtype', [np.uint8, np.float16, np.float32, np.float64])
@pytest.mark.parametrize('order', [1, 3, 5])
def test_nonzero_order_warp_dtype(dtype, order):
    img = _convert(astronaut()[:10, :10, 0], dtype)
    float_dtype = _supported_float_type(dtype)
    assert resize(img, (12, 12), order=order).dtype == float_dtype
    assert rescale(img, 0.5, order=order).dtype == float_dtype
    assert rotate(img, 45, order=order).dtype == float_dtype
    assert warp_polar(img, order=order).dtype == float_dtype
    assert swirl(img, order=order).dtype == float_dtype