import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage.draw import random_shapes
def test_throws_when_intensity_range_out_of_range():
    with testing.raises(ValueError):
        random_shapes((1000, 1234), max_shapes=1, channel_axis=None, intensity_range=(0, 256))
    with testing.raises(ValueError):
        random_shapes((2, 2), max_shapes=1, intensity_range=((-1, 255),))