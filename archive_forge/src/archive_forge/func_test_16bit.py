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
def test_16bit(self):
    image = np.zeros((21, 21), dtype=np.uint16)
    footprint = np.ones((3, 3), dtype=np.uint8)
    for bitdepth in range(17):
        value = 2 ** bitdepth - 1
        image[10, 10] = value
        if bitdepth >= 11:
            expected = ['Bad rank filter performance']
        else:
            expected = []
        with expected_warnings(expected):
            assert rank.minimum(image, footprint)[10, 10] == 0
            assert rank.maximum(image, footprint)[10, 10] == value
            mean_val = rank.mean(image, footprint)[10, 10]
            assert mean_val == int(value / footprint.size)