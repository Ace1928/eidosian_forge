import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal
from skimage.util.shape import view_as_blocks, view_as_windows
def test_view_as_blocks_3D_array():
    A = np.arange(4 * 4 * 6).reshape(4, 4, 6)
    B = view_as_blocks(A, (1, 2, 2))
    assert_equal(B.shape, (4, 2, 3, 1, 2, 2))
    assert_equal(B[2:, 0, 2], np.array([[[[52, 53], [58, 59]]], [[[76, 77], [82, 83]]]]))