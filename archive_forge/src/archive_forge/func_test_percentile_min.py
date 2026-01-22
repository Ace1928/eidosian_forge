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
def test_percentile_min(self):
    img = data.camera()
    img16 = img.astype(np.uint16)
    footprint = disk(15)
    img_p0 = rank.percentile(img, footprint=footprint, p0=0)
    img_min = rank.minimum(img, footprint=footprint)
    assert_equal(img_p0, img_min)
    img_p0 = rank.percentile(img16, footprint=footprint, p0=0)
    img_min = rank.minimum(img16, footprint=footprint)
    assert_equal(img_p0, img_min)