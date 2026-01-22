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
def test_blob_dog(dtype, threshold_type):
    r2 = math.sqrt(2)
    img = np.ones((512, 512), dtype=dtype)
    xs, ys = disk((400, 130), 5)
    img[xs, ys] = 255
    xs, ys = disk((100, 300), 25)
    img[xs, ys] = 255
    xs, ys = disk((200, 350), 45)
    img[xs, ys] = 255
    if threshold_type == 'absolute':
        threshold = 2.0
        if img.dtype.kind != 'f':
            threshold /= np.ptp(img)
        threshold_rel = None
    elif threshold_type == 'relative':
        threshold = None
        threshold_rel = 0.5
    blobs = blob_dog(img, min_sigma=4, max_sigma=50, threshold=threshold, threshold_rel=threshold_rel)

    def radius(x):
        return r2 * x[2]
    s = sorted(blobs, key=radius)
    thresh = 5
    ratio_thresh = 0.25
    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b) - 5) <= ratio_thresh * 5
    b = s[1]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b) - 25) <= ratio_thresh * 25
    b = s[2]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b) - 45) <= ratio_thresh * 45
    img_empty = np.zeros((100, 100), dtype=dtype)
    assert blob_dog(img_empty).size == 0