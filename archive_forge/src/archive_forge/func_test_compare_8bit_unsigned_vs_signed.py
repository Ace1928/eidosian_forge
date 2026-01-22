import inspect
import numpy as np
import pytest
from skimage import data, morphology, util
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import __3Dfilters as _3d_rank_filters
from skimage.filters.rank import subtract_mean
from skimage.morphology import ball, disk, gray
from skimage.util import img_as_float, img_as_ubyte
def test_compare_8bit_unsigned_vs_signed(self):
    image = img_as_ubyte(data.camera())[::2, ::2]
    image[image > 127] = 0
    image_s = image.astype(np.int8)
    image_u = img_as_ubyte(image_s)
    assert_equal(image_u, img_as_ubyte(image_s))
    methods = ['autolevel', 'equalize', 'gradient', 'maximum', 'mean', 'geometric_mean', 'subtract_mean', 'median', 'minimum', 'modal', 'enhance_contrast', 'pop', 'threshold']
    for method in methods:
        func = getattr(rank, method)
        out_u = func(image_u, disk(3))
        with expected_warnings(['Possible precision loss']):
            out_s = func(image_s, disk(3))
        assert_equal(out_u, out_s)