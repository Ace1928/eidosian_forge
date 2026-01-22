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
def test_pass_on_bitdepth(self):
    image = np.full((100, 100), 2 ** 11, dtype=np.uint16)
    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)
    with expected_warnings(['Bad rank filter performance']):
        rank.maximum(image=image, footprint=elem, out=out, mask=mask)