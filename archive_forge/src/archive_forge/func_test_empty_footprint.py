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
def test_empty_footprint(self):
    image = np.zeros((5, 5), dtype=np.uint16)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    res = np.zeros_like(image)
    image[2, 2] = 255
    image[2, 3] = 128
    image[1, 2] = 16
    elem = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.uint8)
    rank.mean(image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0)
    assert_equal(res, out)
    rank.geometric_mean(image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0)
    assert_equal(res, out)
    rank.minimum(image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0)
    assert_equal(res, out)
    rank.maximum(image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0)
    assert_equal(res, out)