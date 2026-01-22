import numpy as np
from skimage.segmentation import clear_border
from skimage._shared.testing import assert_array_equal, assert_
def test_clear_border():
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    result = clear_border(image.copy())
    ref = image.copy()
    ref[1:3, 0:2] = 0
    ref[0:2, -2] = 0
    assert_array_equal(result, ref)
    result = clear_border(image.copy(), 1)
    assert_array_equal(result, np.zeros(result.shape))
    result = clear_border(image.copy(), buffer_size=1, bgval=2)
    assert_array_equal(result, 2 * np.ones_like(image))
    mask = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)
    result = clear_border(image.copy(), mask=mask)
    ref = image.copy()
    ref[1:3, 0:2] = 0
    assert_array_equal(result, ref)