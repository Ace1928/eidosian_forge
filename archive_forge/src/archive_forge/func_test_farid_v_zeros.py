import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
def test_farid_v_zeros():
    """Vertical Farid on an array of all zeros."""
    result = filters.farid_v(np.zeros((10, 10)), mask=np.ones((10, 10), dtype=bool))
    assert_allclose(result, 0, atol=1e-10)