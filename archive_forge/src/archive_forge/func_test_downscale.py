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
@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.float16, np.float32, np.float64])
def test_downscale(dtype):
    x = np.zeros((10, 10), dtype=dtype)
    x[2:4, 2:4] = 1
    scaled = rescale(x, 0.5, order=0, anti_aliasing=False, channel_axis=None, mode='constant')
    expected_dtype = np.float32 if dtype == np.float16 else dtype
    assert scaled.dtype == expected_dtype
    assert scaled.shape == (5, 5)
    assert scaled[1, 1] == 1
    assert scaled[2:, :].sum() == 0
    assert scaled[:, 2:].sum() == 0