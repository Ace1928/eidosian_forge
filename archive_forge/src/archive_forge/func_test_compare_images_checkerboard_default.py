import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage._shared import testing
from skimage.util.compare import compare_images
def test_compare_images_checkerboard_default():
    img1 = np.zeros((2 ** 4, 2 ** 4), dtype=np.uint8)
    img2 = np.full(img1.shape, fill_value=255, dtype=np.uint8)
    res = compare_images(img1, img2, method='checkerboard')
    exp_row1 = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    exp_row2 = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    for i in (0, 1, 4, 5, 8, 9, 12, 13):
        assert_array_equal(res[i, :], exp_row1)
    for i in (2, 3, 6, 7, 10, 11, 14, 15):
        assert_array_equal(res[i, :], exp_row2)