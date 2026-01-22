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
def test_cmyk():
    ref = imread(fetch('data/color.png'))
    img = Image.open(fetch('data/color.png'))
    img = img.convert('CMYK')
    with NamedTemporaryFile(suffix='.jpg') as f:
        fname = f.name
    img.save(fname)
    try:
        img.close()
    except AttributeError:
        pass
    new = imread(fname)
    ref_lab = rgb2lab(ref)
    new_lab = rgb2lab(new)
    for i in range(3):
        newi = np.ascontiguousarray(new_lab[:, :, i])
        refi = np.ascontiguousarray(ref_lab[:, :, i])
        sim = structural_similarity(refi, newi, data_range=refi.max() - refi.min())
        assert sim > 0.99