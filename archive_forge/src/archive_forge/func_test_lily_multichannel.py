from packaging.version import Version
import numpy as np
import skimage.data as data
from skimage.data._fetchers import _image_fetcher
from skimage import io
from skimage._shared.testing import assert_equal, assert_almost_equal, fetch
import os
import pytest
@pytest.mark.xfail(Version(np.__version__) >= Version('2.0.0.dev0'), reason='tifffile uses deprecated attribute `ndarray.newbyteorder`')
def test_lily_multichannel():
    """Test that microscopy image of lily of the valley can be loaded.

    Needs internet connection.
    """
    fetch('data/lily.tif')
    lily = data.lily()
    assert lily.shape == (922, 922, 4)