import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import chan_vese
def test_chan_vese_remove_noise():
    ref = np.zeros((10, 10))
    ref[1:6, 1:6] = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]])
    img = ref.copy()
    img[8, 3] = 1
    result = chan_vese(img, mu=0.3, tol=0.001, max_num_iter=100, dt=10, init_level_set='disk').astype(float)
    assert_array_equal(result, ref)