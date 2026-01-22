import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_roberts_diagonal1(dtype):
    """Roberts' filter on a diagonal edge should be a diagonal line."""
    image = np.tri(10, 10, 0, dtype=dtype)
    expected = ~(np.tri(10, 10, -1).astype(bool) | np.tri(10, 10, -2).astype(bool).transpose())
    expected[-1, -1] = 0
    result = filters.roberts(image)
    assert result.dtype == _supported_float_type(dtype)
    assert_array_almost_equal(result.astype(bool), expected)