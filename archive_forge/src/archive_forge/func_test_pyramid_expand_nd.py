import math
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.transform import pyramids
def test_pyramid_expand_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*(4,) * ndim)
        out = pyramids.pyramid_expand(img, upscale=2, channel_axis=None)
        expected_shape = np.asarray(img.shape) * 2
        assert_array_equal(out.shape, expected_shape)