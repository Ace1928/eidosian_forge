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
def test_isodata_16bit(self):
    np.random.seed(0)
    imfloat = np.random.rand(256, 256)
    assert 0.49 < threshold_isodata(imfloat, nbins=1024) < 0.51
    assert all(0.49 < threshold_isodata(imfloat, nbins=1024, return_all=True))