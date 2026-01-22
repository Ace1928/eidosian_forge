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
def test_blob_dog_3d(dtype, threshold_type):
    r = 10
    pad = 10
    im3 = ellipsoid(r, r, r)
    im3 = np.pad(im3, pad, mode='constant')
    if threshold_type == 'absolute':
        threshold = 0.001
        threshold_rel = 0
    elif threshold_type == 'relative':
        threshold = 0
        threshold_rel = 0.5
    blobs = blob_dog(im3, min_sigma=3, max_sigma=10, sigma_ratio=1.2, threshold=threshold, threshold_rel=threshold_rel)
    b = blobs[0]
    assert b.shape == (4,)
    assert b[0] == r + pad + 1
    assert b[1] == r + pad + 1
    assert b[2] == r + pad + 1
    assert abs(math.sqrt(3) * b[3] - r) < 1.1