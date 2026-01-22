import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal
from skimage.util.shape import view_as_blocks, view_as_windows
def test_view_as_windows_input_not_array():
    A = [1, 2, 3, 4, 5]
    with testing.raises(TypeError):
        view_as_windows(A, (2,))