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
def test_squared_dot():
    im = np.zeros((50, 50))
    im[4:8, 4:8] = 1
    im = img_as_float(im)
    results = peak_local_max(corner_harris(im), min_distance=10, threshold_rel=0)
    assert (results == np.array([[6, 6]])).all()
    results = peak_local_max(corner_shi_tomasi(im), min_distance=10, threshold_rel=0)
    assert (results == np.array([[6, 6]])).all()