import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage import io, draw
from skimage.data import binary_blobs
from skimage.morphology import skeletonize, skeletonize_3d
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_, parametrize, fetch
def test_skeletonize_1D():
    im = np.ones((5, 1), dtype=bool)
    res = skeletonize(im, method='lee')
    assert_equal(res, im)