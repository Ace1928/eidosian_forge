import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data, draw, img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.feature import (
from skimage.morphology import cube, octagon
@run_in_parallel()
def test_corner_fast_astronaut():
    img = rgb2gray(data.astronaut())
    expected = np.array([[444, 310], [374, 171], [249, 171], [492, 139], [403, 162], [496, 266], [362, 328], [476, 250], [353, 172], [346, 279], [494, 169], [177, 156], [413, 181], [213, 117], [390, 149], [140, 205], [232, 266], [489, 155], [387, 195], [101, 198], [363, 192], [364, 147], [300, 244], [325, 245], [141, 242], [401, 197], [197, 148], [339, 242], [188, 113], [362, 252], [379, 183], [358, 307], [245, 137], [369, 159], [464, 251], [305, 57], [223, 375]])
    actual = corner_peaks(corner_fast(img, 12, 0.3), min_distance=10, threshold_rel=0)
    assert_array_equal(actual, expected)