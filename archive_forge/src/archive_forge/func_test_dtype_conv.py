import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage import io, draw
from skimage.data import binary_blobs
from skimage.morphology import skeletonize, skeletonize_3d
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_, parametrize, fetch
def test_dtype_conv():
    img = np.random.random((16, 16))[::2, ::2]
    img[img < 0.5] = 0
    orig = img.copy()
    res = skeletonize(img, method='lee')
    assert res.dtype == bool
    assert_equal(img, orig)