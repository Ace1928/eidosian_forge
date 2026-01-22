import pathlib
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from skimage._shared.testing import fetch
from skimage.io import imread, imsave, reset_plugins, use_plugin
def test_imread_handle():
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    with open(fetch('data/chessboard_GRAY_U16.tif'), 'rb') as fh:
        img = imread(fh)
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)