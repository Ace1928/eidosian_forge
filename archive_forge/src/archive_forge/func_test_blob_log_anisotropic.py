import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
def test_blob_log_anisotropic():
    image = np.zeros((50, 50))
    image[20, 10:20] = 1
    isotropic_blobs = blob_log(image, min_sigma=0.5, max_sigma=2, num_sigma=3)
    assert len(isotropic_blobs) > 1
    ani_blobs = blob_log(image, min_sigma=[0.5, 5], max_sigma=[2, 20], num_sigma=3)
    assert len(ani_blobs) == 1