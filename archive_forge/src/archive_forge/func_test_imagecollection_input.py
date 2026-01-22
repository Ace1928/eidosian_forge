import os
import itertools
import numpy as np
import imageio.v3 as iio3
from skimage import data_dir
from skimage.io.collection import ImageCollection, MultiImage, alphanumeric_key
from skimage.io import reset_plugins
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_allclose, fetch
import pytest
def test_imagecollection_input():
    """Test function for ImageCollection. The new behavior (implemented
    in 0.16) allows the `pattern` argument to accept a list of strings
    as the input.

    Notes
    -----
        If correct, `images` will receive three images.
    """
    pics = [fetch('data/coffee.png'), fetch('data/chessboard_GRAY.png'), fetch('data/rocket.jpg')]
    pattern = [os.path.join(data_dir, pic) for pic in pics]
    images = ImageCollection(pattern)
    assert len(images) == 3