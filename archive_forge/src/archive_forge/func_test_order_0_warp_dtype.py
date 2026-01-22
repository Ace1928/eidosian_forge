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
@pytest.mark.parametrize('dtype', [np.uint8, bool, np.float32, np.float64])
def test_order_0_warp_dtype(dtype):
    img = _convert(astronaut()[:10, :10, 0], dtype)
    assert resize(img, (12, 12), order=0).dtype == dtype
    assert rescale(img, 0.5, order=0).dtype == dtype
    assert rotate(img, 45, order=0).dtype == dtype
    assert warp_polar(img, order=0).dtype == dtype
    assert swirl(img, order=0).dtype == dtype