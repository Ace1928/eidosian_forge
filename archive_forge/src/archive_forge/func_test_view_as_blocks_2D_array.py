import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal
from skimage.util.shape import view_as_blocks, view_as_windows
def test_view_as_blocks_2D_array():
    A = np.arange(4 * 4).reshape(4, 4)
    B = view_as_blocks(A, (2, 2))
    assert_equal(B[0, 1], np.array([[2, 3], [6, 7]]))
    assert_equal(B[1, 0, 1, 1], 13)