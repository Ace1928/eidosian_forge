import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import chan_vese
def test_chan_vese_incorrect_image_type():
    img = np.zeros((10, 10, 3))
    ls = np.zeros((10, 9))
    with pytest.raises(ValueError):
        chan_vese(img, mu=0.0, init_level_set=ls)