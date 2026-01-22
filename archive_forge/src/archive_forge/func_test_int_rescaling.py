import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
@pytest.mark.parametrize('function_name', ['farid', 'laplace', 'prewitt', 'roberts', 'scharr', 'sobel'])
def test_int_rescaling(function_name):
    """Basic test that uint8 inputs get rescaled from [0, 255] to [0, 1.]

    The output of any of these filters should be within roughly a factor of
    two of the input range. For integer inputs, rescaling to floats in
    [0.0, 1.0] should occur, so just verify outputs are not > 2.0.
    """
    img = data.coins()[:128, :128]
    func = getattr(filters, function_name)
    filtered = func(img)
    assert filtered.max() <= 2.0