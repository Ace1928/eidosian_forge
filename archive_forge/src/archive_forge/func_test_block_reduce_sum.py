import numpy as np
from skimage.measure import block_reduce
from skimage._shared import testing
from skimage._shared.testing import assert_equal
def test_block_reduce_sum():
    image1 = np.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3))
    expected1 = np.array([[24, 42], [96, 114]])
    assert_equal(expected1, out1)
    image2 = np.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (3, 3))
    expected2 = np.array([[81, 108, 87], [174, 192, 138]])
    assert_equal(expected2, out2)