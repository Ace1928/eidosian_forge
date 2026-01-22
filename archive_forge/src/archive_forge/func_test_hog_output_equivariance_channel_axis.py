import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
@pytest.mark.parametrize('channel_axis', [0, 1, -1, -2])
def test_hog_output_equivariance_channel_axis(channel_axis):
    img = data.astronaut()[:64, :32]
    img[:, :, (1, 2)] = 0
    img = np.moveaxis(img, -1, channel_axis)
    hog_ref = feature.hog(img, channel_axis=channel_axis, block_norm='L1')
    for n in (1, 2):
        hog_fact = feature.hog(np.roll(img, n, axis=channel_axis), channel_axis=channel_axis, block_norm='L1')
        assert_almost_equal(hog_ref, hog_fact)