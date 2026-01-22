import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
@pytest.mark.parametrize('grad_func', (filters.prewitt_h, filters.sobel_h, filters.scharr_h))
def test_horizontal_mask_line(grad_func):
    """Horizontal edge filters mask pixels surrounding input mask."""
    vgrad, _ = np.mgrid[:1:11j, :1:11j]
    vgrad[5, :] = 1
    mask = np.ones_like(vgrad)
    mask[5, :] = 0
    expected = np.zeros_like(vgrad)
    expected[1:-1, 1:-1] = 0.2
    expected[4:7, 1:-1] = 0
    result = grad_func(vgrad, mask)
    assert_allclose(result, expected)