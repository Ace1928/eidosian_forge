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
@pytest.fixture
def im3d():
    r = 10
    pad = 10
    im3 = draw.ellipsoid(r, r, r)
    im3 = np.pad(im3, pad, mode='constant').astype(np.uint8)
    return im3