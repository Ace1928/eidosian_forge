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
def test_keep_range():
    image = np.linspace(0, 2, 25).reshape(5, 5)
    out = rescale(image, 2, preserve_range=False, clip=True, order=0, mode='constant', channel_axis=None, anti_aliasing=False)
    assert out.min() == 0
    assert out.max() == 2
    out = rescale(image, 2, preserve_range=True, clip=True, order=0, mode='constant', channel_axis=None, anti_aliasing=False)
    assert out.min() == 0
    assert out.max() == 2
    out = rescale(image.astype(np.uint8), 2, preserve_range=False, mode='constant', channel_axis=None, anti_aliasing=False, clip=True, order=0)
    assert out.min() == 0
    assert out.max() == 2