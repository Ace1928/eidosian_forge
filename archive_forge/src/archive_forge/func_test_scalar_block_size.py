import numpy as np
from skimage.measure import block_reduce
from skimage._shared import testing
from skimage._shared.testing import assert_equal
def test_scalar_block_size():
    image = np.arange(6 * 6).reshape(6, 6)
    out = block_reduce(image, 3, func=np.min)
    expected1 = np.array([[0, 3], [18, 21]])
    assert_equal(expected1, out)
    expected2 = block_reduce(image, (3, 3), func=np.min)
    assert_equal(expected2, out)