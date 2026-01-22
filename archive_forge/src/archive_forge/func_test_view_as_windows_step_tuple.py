import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal
from skimage.util.shape import view_as_blocks, view_as_windows
def test_view_as_windows_step_tuple():
    A = np.arange(24).reshape((6, 4))
    B = view_as_windows(A, (3, 2), step=3)
    assert B.shape == (2, 1, 3, 2)
    assert B.size != A.size
    C = view_as_windows(A, (3, 2), step=(3, 2))
    assert C.shape == (2, 2, 3, 2)
    assert C.size == A.size
    assert_equal(C, [[[[0, 1], [4, 5], [8, 9]], [[2, 3], [6, 7], [10, 11]]], [[[12, 13], [16, 17], [20, 21]], [[14, 15], [18, 19], [22, 23]]]])