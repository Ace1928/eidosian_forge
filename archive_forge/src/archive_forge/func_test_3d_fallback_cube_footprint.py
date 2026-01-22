import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
@pytest.mark.parametrize('function', gray_3d_fallback_functions)
def test_3d_fallback_cube_footprint(function):
    image = np.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1
    cube = np.ones((3, 3, 3), dtype=np.uint8)
    new_image = function(image, cube)
    assert_array_equal(new_image, image)