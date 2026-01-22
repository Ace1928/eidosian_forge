from tempfile import NamedTemporaryFile
import numpy as np
from skimage import io
from skimage.io import imread, imsave, use_plugin, reset_plugins
from skimage._shared import testing
from skimage._shared.testing import (
from pytest import importorskip
importorskip('imread')
def test_bilevel():
    expected = np.zeros((10, 10), bool)
    expected[::2] = 1
    img = imread(fetch('data/checker_bilevel.png'))
    assert_array_equal(img.astype(bool), expected)