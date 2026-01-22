import math
import numpy as np
import pytest
from numpy.testing import (
from scipy import ndimage as ndi
from skimage import data, util
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.draw import disk
from skimage.exposure import histogram
from skimage.filters._multiotsu import (
from skimage.filters.thresholding import (
def test_isodata_moon_image():
    moon = util.img_as_ubyte(data.moon())
    threshold = threshold_isodata(moon)
    assert np.floor((moon[moon <= threshold].mean() + moon[moon > threshold].mean()) / 2.0) == threshold
    assert threshold == 86
    thresholds = threshold_isodata(moon, return_all=True)
    for threshold in thresholds:
        assert np.floor((moon[moon <= threshold].mean() + moon[moon > threshold].mean()) / 2.0) == threshold
    assert_equal(thresholds, [86, 87, 88, 122, 123, 124, 139, 140])