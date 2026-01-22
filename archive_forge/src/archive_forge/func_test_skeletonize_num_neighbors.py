import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage import io, draw
from skimage.data import binary_blobs
from skimage.morphology import skeletonize, skeletonize_3d
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_, parametrize, fetch
@pytest.mark.parametrize('dtype', [bool, float, int])
def test_skeletonize_num_neighbors(dtype):
    image = np.zeros((300, 300), dtype=dtype)
    image[10:-10, 10:100] = 1
    image[-100:-10, 10:-10] = 2
    image[10:-10, -100:-10] = 3
    rs, cs = draw.line(250, 150, 10, 280)
    for i in range(10):
        image[rs + i, cs] = 4
    rs, cs = draw.line(10, 150, 250, 280)
    for i in range(20):
        image[rs + i, cs] = 5
    ir, ic = np.indices(image.shape)
    circle1 = (ic - 135) ** 2 + (ir - 150) ** 2 < 30 ** 2
    circle2 = (ic - 135) ** 2 + (ir - 150) ** 2 < 20 ** 2
    image[circle1] = 1
    image[circle2] = 0
    result = skeletonize(image, method='lee').astype(np.uint8)
    mask = np.array([[1, 1], [1, 1]], np.uint8)
    blocks = ndi.correlate(result, mask, mode='constant')
    assert_(not np.any(blocks == 4))