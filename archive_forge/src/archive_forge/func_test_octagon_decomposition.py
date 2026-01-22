import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
@pytest.mark.parametrize('function', ['erosion', 'dilation', 'closing', 'opening', 'white_tophat', 'black_tophat'])
@pytest.mark.parametrize('m', (0, 1, 3, 5))
@pytest.mark.parametrize('n', (0, 1, 2, 3))
@pytest.mark.parametrize('decomposition', ['sequence'])
def test_octagon_decomposition(cam_image, function, m, n, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    if m == 0 and n == 0:
        with pytest.raises(ValueError):
            footprints.octagon(m, n, decomposition=decomposition)
    else:
        footprint_ndarray = footprints.octagon(m, n, decomposition=None)
        footprint = footprints.octagon(m, n, decomposition=decomposition)
        func = getattr(gray, function)
        expected = func(cam_image, footprint=footprint_ndarray)
        out = func(cam_image, footprint=footprint)
        assert_array_equal(expected, out)