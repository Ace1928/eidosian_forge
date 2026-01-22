import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import find_boundaries, mark_boundaries
@pytest.mark.parametrize('dtype', [np.uint8, np.float16, np.float32, np.float64])
def test_mark_boundaries(dtype):
    image = np.zeros((10, 10), dtype=dtype)
    label_image = np.zeros((10, 10), dtype=np.uint8)
    label_image[2:7, 2:7] = 1
    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    marked = mark_boundaries(image, label_image, color=white, mode='thick')
    assert marked.dtype == _supported_float_type(dtype)
    result = np.mean(marked, axis=-1)
    assert_array_equal(result, ref)
    ref = np.array([[0, 2, 2, 2, 2, 2, 2, 2, 0, 0], [2, 2, 1, 1, 1, 1, 1, 2, 2, 0], [2, 1, 1, 1, 1, 1, 1, 1, 2, 0], [2, 1, 1, 2, 2, 2, 1, 1, 2, 0], [2, 1, 1, 2, 0, 2, 1, 1, 2, 0], [2, 1, 1, 2, 2, 2, 1, 1, 2, 0], [2, 1, 1, 1, 1, 1, 1, 1, 2, 0], [2, 2, 1, 1, 1, 1, 1, 2, 2, 0], [0, 2, 2, 2, 2, 2, 2, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    marked = mark_boundaries(image, label_image, color=white, outline_color=(2, 2, 2), mode='thick')
    result = np.mean(marked, axis=-1)
    assert_array_equal(result, ref)