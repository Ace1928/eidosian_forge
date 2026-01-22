import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import chan_vese
def test_chan_vese_small_disk_level_set():
    img = np.zeros((10, 10))
    img[3:6, 3:6] = 1
    result = chan_vese(img, mu=0.0, tol=0.001, init_level_set='small disk')
    assert_array_equal(result.astype(float), img)