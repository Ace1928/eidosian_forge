import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
def test_farid_horizontal():
    """Farid on a horizontal edge should be a horizontal line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.farid(image) * np.sqrt(2)
    assert np.all(result[i == 0] == result[i == 0][0])
    assert_allclose(result[np.abs(i) > 2], 0, atol=1e-10)