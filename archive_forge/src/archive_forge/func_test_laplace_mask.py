import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_laplace_mask(dtype):
    """Laplace on a masked array should be zero."""
    image = np.zeros((9, 9), dtype=dtype)
    image[3:-3, 3:-3] = 1
    result = filters.laplace(image, ksize=3, mask=np.zeros((9, 9), dtype=bool))
    assert result.dtype == _supported_float_type(dtype)
    assert np.all(result == 0)