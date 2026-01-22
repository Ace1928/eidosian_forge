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
def test_imsave_incorrect_dimension():
    with temporary_file(suffix='.png') as fname:
        with testing.raises(ValueError):
            with expected_warnings([fname + ' is a low contrast image']):
                imsave(fname, np.zeros((2, 3, 3, 1)))
        with testing.raises(ValueError):
            with expected_warnings([fname + ' is a low contrast image']):
                imsave(fname, np.zeros((2, 3, 2)))
        with testing.raises(ValueError):
            with expected_warnings([]):
                imsave(fname, np.zeros((2, 3, 2)), check_contrast=False)