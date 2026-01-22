import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
def test_hog_output_equivariance_deprecated_multichannel():
    img = data.astronaut()
    img[:, :, (1, 2)] = 0
    hog_ref = feature.hog(img, channel_axis=-1, block_norm='L1')
    for n in (1, 2):
        hog_fact = feature.hog(np.roll(img, n, axis=2), channel_axis=-1, block_norm='L1')
        assert_almost_equal(hog_ref, hog_fact)