import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
def test_scharr_h_mask():
    """Horizontal Scharr on a masked array should be zero."""
    result = filters.scharr_h(np.random.uniform(size=(10, 10)), np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)