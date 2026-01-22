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
def test_isodata_camera_image():
    camera = util.img_as_ubyte(data.camera())
    threshold = threshold_isodata(camera)
    assert np.floor((camera[camera <= threshold].mean() + camera[camera > threshold].mean()) / 2.0) == threshold
    assert threshold == 102
    assert (threshold_isodata(camera, return_all=True) == [102, 103]).all()