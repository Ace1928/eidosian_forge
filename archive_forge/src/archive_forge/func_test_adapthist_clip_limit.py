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
def test_adapthist_clip_limit():
    img_u = data.moon()
    img_f = util.img_as_float(img_u)
    img_clahe0 = exposure.equalize_adapthist(img_u, clip_limit=0)
    img_clahe1 = exposure.equalize_adapthist(img_u, clip_limit=1)
    assert_array_equal(img_clahe0, img_clahe1)
    img_clahe0 = exposure.equalize_adapthist(img_f, clip_limit=0)
    img_clahe1 = exposure.equalize_adapthist(img_f, clip_limit=1)
    assert_array_equal(img_clahe0, img_clahe1)