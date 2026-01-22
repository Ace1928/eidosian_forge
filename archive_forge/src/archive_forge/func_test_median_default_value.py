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
def test_median_default_value(self):
    a = np.zeros((3, 3), dtype=np.uint8)
    a[1] = 1
    full_footprint = np.ones((3, 3), dtype=np.uint8)
    assert_equal(rank.median(a), rank.median(a, full_footprint))
    assert rank.median(a)[1, 1] == 0
    assert rank.median(a, disk(1))[1, 1] == 1