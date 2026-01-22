import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
def test_blob_doh_overlap():
    img = np.ones((256, 256), dtype=np.uint8)
    xs, ys = disk((100, 100), 20)
    img[xs, ys] = 255
    xs, ys = disk((120, 100), 30)
    img[xs, ys] = 255
    blobs = blob_doh(img, min_sigma=1, max_sigma=60, num_sigma=10, threshold=0.05)
    assert len(blobs) == 1