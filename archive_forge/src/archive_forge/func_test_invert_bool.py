import numpy as np
from skimage import dtype_limits
from skimage.util.dtype import dtype_range
from skimage.util import invert
from skimage._shared.testing import assert_array_equal
def test_invert_bool():
    dtype = 'bool'
    image = np.zeros((3, 3), dtype=dtype)
    upper_dtype_limit = dtype_limits(image, clip_negative=False)[1]
    image[1, :] = upper_dtype_limit
    expected = np.zeros((3, 3), dtype=dtype) + upper_dtype_limit
    expected[1, :] = 0
    result = invert(image)
    assert_array_equal(expected, result)