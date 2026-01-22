import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
def test_farid_h_vertical():
    """Horizontal Farid on a vertical edge should be zero."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float) * np.sqrt(2)
    result = filters.farid_h(image)
    assert_allclose(result, 0, atol=1e-10)