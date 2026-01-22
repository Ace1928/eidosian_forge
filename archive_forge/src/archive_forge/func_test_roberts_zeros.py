import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_roberts_zeros(dtype):
    """Roberts' filter on an array of all zeros."""
    result = filters.roberts(np.zeros((10, 10), dtype=dtype), np.ones((10, 10), bool))
    assert result.dtype == _supported_float_type(dtype)
    assert np.all(result == 0)