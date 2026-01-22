from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_array_equal
import numpy as np
from skimage.util import montage
def test_montage_simple_padding_gray():
    n_images, n_rows, n_cols = (2, 2, 2)
    arr_in = np.arange(n_images * n_rows * n_cols)
    arr_in = arr_in.reshape(n_images, n_rows, n_cols)
    arr_out = montage(arr_in, padding_width=1)
    arr_ref = np.array([[3, 3, 3, 3, 3, 3, 3], [3, 0, 1, 3, 4, 5, 3], [3, 2, 3, 3, 6, 7, 3], [3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3]])
    assert_array_equal(arr_out, arr_ref)