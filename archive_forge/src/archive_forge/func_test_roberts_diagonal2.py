import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
def test_roberts_diagonal2():
    """Roberts' filter on a diagonal edge should be a diagonal line."""
    image = np.rot90(np.tri(10, 10, 0), 3)
    expected = ~np.rot90(np.tri(10, 10, -1).astype(bool) | np.tri(10, 10, -2).astype(bool).transpose())
    expected = _mask_filter_result(expected, None)
    result = filters.roberts(image).astype(bool)
    assert_array_almost_equal(result, expected)