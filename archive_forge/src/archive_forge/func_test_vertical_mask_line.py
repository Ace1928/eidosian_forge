import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
@pytest.mark.parametrize('grad_func', (filters.prewitt_v, filters.sobel_v, filters.scharr_v))
def test_vertical_mask_line(grad_func):
    """Vertical edge filters mask pixels surrounding input mask."""
    _, hgrad = np.mgrid[:1:11j, :1:11j]
    hgrad[:, 5] = 1
    mask = np.ones_like(hgrad)
    mask[:, 5] = 0
    expected = np.zeros_like(hgrad)
    expected[1:-1, 1:-1] = 0.2
    expected[1:-1, 4:7] = 0
    result = grad_func(hgrad, mask)
    assert_allclose(result, expected)