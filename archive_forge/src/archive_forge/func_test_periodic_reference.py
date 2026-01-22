import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_periodic_reference(dtype):
    img = data.astronaut()
    img = rgb2gray(img)
    s = np.linspace(0, 2 * np.pi, 400)
    r = 100 + 100 * np.sin(s)
    c = 220 + 100 * np.cos(s)
    init = np.array([r, c]).T
    img_smooth = gaussian(img, sigma=3, preserve_range=False).astype(dtype, copy=False)
    snake = active_contour(img_smooth, init, alpha=0.015, beta=10, w_line=0, w_edge=1, gamma=0.001)
    assert snake.dtype == _supported_float_type(dtype)
    refr = [98, 99, 100, 101, 102, 103, 104, 105, 106, 108]
    refc = [299, 298, 298, 298, 298, 297, 297, 296, 296, 295]
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)