import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import chan_vese
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_chan_vese_flat_level_set(dtype):
    img = np.zeros((10, 10), dtype=dtype)
    img[3:6, 3:6] = 1
    ls = np.full((10, 10), 1000, dtype=dtype)
    result = chan_vese(img, mu=0.0, tol=0.001, init_level_set=ls)
    assert_array_equal(result.astype(float), np.ones((10, 10)))
    result = chan_vese(img, mu=0.0, tol=0.001, init_level_set=-ls)
    assert_array_equal(result.astype(float), np.zeros((10, 10)))