import os
from io import BytesIO
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
from PIL import Image
from skimage._shared import testing
from skimage._shared._tempfile import temporary_file
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.metrics import structural_similarity
from ... import img_as_float
from ...color import rgb2lab
from .. import imread, imsave, reset_plugins, use_plugin, plugin_order
from .._plugins.pil_plugin import _palette_is_grayscale, ndarray_to_pil, pil_to_ndarray
def test_multi_page_gif():
    img = imread(fetch('data/no_time_for_that_tiny.gif'))
    assert img.shape == (24, 25, 14, 3), img.shape
    img2 = imread(fetch('data/no_time_for_that_tiny.gif'), img_num=5)
    assert img2.shape == (25, 14, 3)
    assert_allclose(img[5], img2)