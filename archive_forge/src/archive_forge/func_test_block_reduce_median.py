import numpy as np
from skimage.measure import block_reduce
from skimage._shared import testing
from skimage._shared.testing import assert_equal
def test_block_reduce_median():
    image1 = np.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3), func=np.median)
    expected1 = np.array([[4.0, 7.0], [16.0, 19.0]])
    assert_equal(expected1, out1)
    image2 = np.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (4, 5), func=np.median)
    expected2 = np.array([[14.0, 6.5], [0.0, 0.0]])
    assert_equal(expected2, out2)
    image3 = np.array([[1, 5, 5, 5], [5, 5, 5, 1000]])
    out3 = block_reduce(image3, (2, 4), func=np.median)
    assert_equal(5, out3)