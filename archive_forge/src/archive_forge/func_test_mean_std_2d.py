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
@pytest.mark.parametrize('window_size, mean_kernel', [(11, np.full((11,) * 2, 1 / 11 ** 2)), ((11, 11), np.full((11, 11), 1 / 11 ** 2)), ((9, 13), np.full((9, 13), 1 / math.prod((9, 13)))), ((13, 9), np.full((13, 9), 1 / math.prod((13, 9)))), ((1, 9), np.full((1, 9), 1 / math.prod((1, 9))))])
def test_mean_std_2d(window_size, mean_kernel):
    image = np.random.rand(256, 256)
    m, s = _mean_std(image, w=window_size)
    expected_m = ndi.convolve(image, mean_kernel, mode='mirror')
    assert_allclose(m, expected_m)
    expected_s = ndi.generic_filter(image, np.std, size=window_size, mode='mirror')
    assert_allclose(s, expected_s)