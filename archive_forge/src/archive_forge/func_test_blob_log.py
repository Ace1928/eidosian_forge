import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
@pytest.mark.parametrize('dtype', [np.uint8, np.float16, np.float32, np.float64])
@pytest.mark.parametrize('threshold_type', ['absolute', 'relative'])
def test_blob_log(dtype, threshold_type):
    r2 = math.sqrt(2)
    img = np.ones((256, 256), dtype=dtype)
    xs, ys = disk((200, 65), 5)
    img[xs, ys] = 255
    xs, ys = disk((80, 25), 15)
    img[xs, ys] = 255
    xs, ys = disk((50, 150), 25)
    img[xs, ys] = 255
    xs, ys = disk((100, 175), 30)
    img[xs, ys] = 255
    if threshold_type == 'absolute':
        threshold = 1
        if img.dtype.kind != 'f':
            threshold /= np.ptp(img)
        threshold_rel = None
    elif threshold_type == 'relative':
        threshold = None
        threshold_rel = 0.5
    blobs = blob_log(img, min_sigma=5, max_sigma=20, threshold=threshold, threshold_rel=threshold_rel)

    def radius(x):
        return r2 * x[2]
    s = sorted(blobs, key=radius)
    thresh = 3
    b = s[0]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 65) <= thresh
    assert abs(radius(b) - 5) <= thresh
    b = s[1]
    assert abs(b[0] - 80) <= thresh
    assert abs(b[1] - 25) <= thresh
    assert abs(radius(b) - 15) <= thresh
    b = s[2]
    assert abs(b[0] - 50) <= thresh
    assert abs(b[1] - 150) <= thresh
    assert abs(radius(b) - 25) <= thresh
    b = s[3]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 175) <= thresh
    assert abs(radius(b) - 30) <= thresh
    blobs = blob_log(img, min_sigma=5, max_sigma=20, threshold=threshold, threshold_rel=threshold_rel, log_scale=True)
    b = s[0]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 65) <= thresh
    assert abs(radius(b) - 5) <= thresh
    b = s[1]
    assert abs(b[0] - 80) <= thresh
    assert abs(b[1] - 25) <= thresh
    assert abs(radius(b) - 15) <= thresh
    b = s[2]
    assert abs(b[0] - 50) <= thresh
    assert abs(b[1] - 150) <= thresh
    assert abs(radius(b) - 25) <= thresh
    b = s[3]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 175) <= thresh
    assert abs(radius(b) - 30) <= thresh
    img_empty = np.zeros((100, 100))
    assert blob_log(img_empty).size == 0