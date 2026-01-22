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
def test_compare_autolevels_16bit(self):
    image = data.camera().astype(np.uint16) * 4
    footprint = disk(20)
    loc_autolevel = rank.autolevel(image, footprint=footprint)
    loc_perc_autolevel = rank.autolevel_percentile(image, footprint=footprint, p0=0.0, p1=1.0)
    assert_equal(loc_autolevel, loc_perc_autolevel)