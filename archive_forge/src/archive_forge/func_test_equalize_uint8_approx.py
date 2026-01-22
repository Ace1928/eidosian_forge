import warnings
import numpy as np
import pytest
from numpy.testing import (
from packaging.version import Version
from skimage import data
from skimage import exposure
from skimage import util
from skimage.color import rgb2gray
from skimage.exposure.exposure import intensity_range
from skimage.util.dtype import dtype_range
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
def test_equalize_uint8_approx():
    """Check integer bins used for uint8 images."""
    img_eq0 = exposure.equalize_hist(test_img_int)
    img_eq1 = exposure.equalize_hist(test_img_int, nbins=3)
    assert_allclose(img_eq0, img_eq1)