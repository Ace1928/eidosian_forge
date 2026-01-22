import numpy as np
from skimage.measure import block_reduce
from skimage._shared import testing
from skimage._shared.testing import assert_equal
def test_block_reduce_min():
    image1 = np.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3), func=np.min)
    expected1 = np.array([[0, 3], [12, 15]])
    assert_equal(expected1, out1)
    image2 = np.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (4, 5), func=np.min)
    expected2 = np.array([[0, 0], [0, 0]])
    assert_equal(expected2, out2)