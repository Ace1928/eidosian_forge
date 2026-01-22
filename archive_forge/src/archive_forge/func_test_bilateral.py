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
def test_bilateral(self):
    image = np.zeros((21, 21), dtype=np.uint16)
    footprint = np.ones((3, 3), dtype=np.uint8)
    image[10, 10] = 1000
    image[10, 11] = 1010
    image[10, 9] = 900
    kwargs = dict(s0=1, s1=1)
    assert rank.mean_bilateral(image, footprint, **kwargs)[10, 10] == 1000
    assert rank.pop_bilateral(image, footprint, **kwargs)[10, 10] == 1
    kwargs = dict(s0=11, s1=11)
    assert rank.mean_bilateral(image, footprint, **kwargs)[10, 10] == 1005
    assert rank.pop_bilateral(image, footprint, **kwargs)[10, 10] == 2