from packaging.version import Version
import numpy as np
import skimage.data as data
from skimage.data._fetchers import _image_fetcher
from skimage import io
from skimage._shared.testing import assert_equal, assert_almost_equal, fetch
import os
import pytest
def test_eagle():
    """Test that "eagle" image can be loaded."""
    fetch('data/eagle.png')
    eagle = data.eagle()
    assert_equal(eagle.ndim, 2)
    assert_equal(eagle.dtype, np.dtype('uint8'))