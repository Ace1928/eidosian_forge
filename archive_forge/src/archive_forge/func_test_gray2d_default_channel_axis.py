from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_gray2d_default_channel_axis():
    img = np.zeros((20, 21))
    img[:10, :10] = 0.33
    with pytest.raises(ValueError, match='channel_axis=-1 indicates multichannel'):
        slic(img)
    slic(img, channel_axis=None)