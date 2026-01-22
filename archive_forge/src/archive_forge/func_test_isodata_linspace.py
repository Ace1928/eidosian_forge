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
def test_isodata_linspace(self):
    image = np.linspace(-127, 0, 256)
    assert -63.8 < threshold_isodata(image) < -63.6
    assert_almost_equal(threshold_isodata(image, return_all=True), [-63.74804688, -63.25195312])