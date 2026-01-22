from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_array_equal
import numpy as np
from skimage.util import montage
def test_montage_rescale_intensity_gray():
    n_images, n_rows, n_cols = (4, 3, 3)
    arr_in = np.arange(n_images * n_rows * n_cols, dtype=np.float32)
    arr_in = arr_in.reshape(n_images, n_rows, n_cols)
    arr_out = montage(arr_in, rescale_intensity=True)
    arr_ref = np.array([[0.0, 0.125, 0.25, 0.0, 0.125, 0.25], [0.375, 0.5, 0.625, 0.375, 0.5, 0.625], [0.75, 0.875, 1.0, 0.75, 0.875, 1.0], [0.0, 0.125, 0.25, 0.0, 0.125, 0.25], [0.375, 0.5, 0.625, 0.375, 0.5, 0.625], [0.75, 0.875, 1.0, 0.75, 0.875, 1.0]])
    assert_equal(arr_out.min(), 0.0)
    assert_equal(arr_out.max(), 1.0)
    assert_array_equal(arr_out, arr_ref)