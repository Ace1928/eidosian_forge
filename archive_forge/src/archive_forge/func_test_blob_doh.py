import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
@pytest.mark.parametrize('dtype', [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize('threshold_type', ['absolute', 'relative'])
def test_blob_doh(dtype, threshold_type):
    img = np.ones((512, 512), dtype=dtype)
    xs, ys = disk((400, 130), 20)
    img[xs, ys] = 255
    xs, ys = disk((460, 50), 30)
    img[xs, ys] = 255
    xs, ys = disk((100, 300), 40)
    img[xs, ys] = 255
    xs, ys = disk((200, 350), 50)
    img[xs, ys] = 255
    if threshold_type == 'absolute':
        threshold = 0.05
        if img.dtype.kind == 'f':
            ptp = np.ptp(img)
            threshold *= ptp ** 2
        threshold_rel = None
    elif threshold_type == 'relative':
        threshold = None
        threshold_rel = 0.5
    blobs = blob_doh(img, min_sigma=1, max_sigma=60, num_sigma=10, threshold=threshold, threshold_rel=threshold_rel)

    def radius(x):
        return x[2]
    s = sorted(blobs, key=radius)
    thresh = 4
    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b) - 20) <= thresh
    b = s[1]
    assert abs(b[0] - 460) <= thresh
    assert abs(b[1] - 50) <= thresh
    assert abs(radius(b) - 30) <= thresh
    b = s[2]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b) - 40) <= thresh
    b = s[3]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b) - 50) <= thresh