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
def test_threshold_niblack_iterable_window_size(self):
    ref = np.array([[False, False, False, True, True], [False, False, True, True, True], [False, True, True, True, False], [False, True, True, True, False], [True, True, False, False, False]])
    thres = threshold_niblack(self.image, window_size=[3, 5], k=0.5)
    out = self.image > thres
    assert_array_equal(ref, out)