import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal
from skimage.util.shape import view_as_blocks, view_as_windows
def test_view_as_windows_2D():
    A = np.arange(5 * 4).reshape(5, 4)
    window_shape = (4, 3)
    B = view_as_windows(A, window_shape)
    assert_equal(B.shape, (2, 2, 4, 3))
    assert_equal(B, np.array([[[[0, 1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14]], [[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15]]], [[[4, 5, 6], [8, 9, 10], [12, 13, 14], [16, 17, 18]], [[5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19]]]]))