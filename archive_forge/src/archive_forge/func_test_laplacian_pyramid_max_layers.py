import math
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.transform import pyramids
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2, -3])
def test_laplacian_pyramid_max_layers(channel_axis):
    for downscale in [2, 3, 5, 7]:
        if channel_axis is None:
            shape = (32, 8)
            shape_without_channels = shape
        else:
            shape_without_channels = (32, 8)
            ndim = len(shape_without_channels) + 1
            n_channels = 5
            shape = list(shape_without_channels)
            shape.insert(channel_axis % ndim, n_channels)
            shape = tuple(shape)
        img = np.ones(shape)
        pyramid = pyramids.pyramid_laplacian(img, downscale=downscale, channel_axis=channel_axis)
        max_layer = math.ceil(math.log(max(shape_without_channels), downscale))
        for layer, out in enumerate(pyramid):
            if channel_axis is None:
                out_shape_without_channels = out.shape
            else:
                assert out.shape[channel_axis] == n_channels
                out_shape_without_channels = list(out.shape)
                out_shape_without_channels.pop(channel_axis)
                out_shape_without_channels = tuple(out_shape_without_channels)
            if layer < max_layer:
                assert max(out_shape_without_channels) > 1
        assert_equal(max_layer, layer)
        assert out_shape_without_channels == (1, 1)