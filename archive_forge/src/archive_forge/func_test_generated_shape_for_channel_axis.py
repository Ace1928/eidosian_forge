import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage.draw import random_shapes
@pytest.mark.parametrize('channel_axis', [None, 0, 1, 2])
def test_generated_shape_for_channel_axis(channel_axis):
    shape = (128, 64)
    num_channels = 5
    image, _ = random_shapes(shape, num_channels=num_channels, min_shapes=3, max_shapes=10, channel_axis=channel_axis)
    if channel_axis is None:
        expected_shape = shape
    else:
        expected_shape = tuple(np.insert(shape, channel_axis, num_channels))
    assert image.shape == expected_shape