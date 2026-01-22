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
def test_adapthist_grayscale_Nd():
    """
    Test for n-dimensional consistency with float images
    Note: Currently if img.ndim == 3, img.shape[2] > 4 must hold for the image
    not to be interpreted as a color image by @adapt_rgb
    """
    img = util.img_as_float(data.astronaut())
    img = rgb2gray(img)
    a = 15
    img2d = util.img_as_float(img[0:-1:a, 0:-1:a])
    img3d = np.array([img2d] * (img.shape[0] // a))
    adapted2d = exposure.equalize_adapthist(img2d, kernel_size=5, clip_limit=0.05)
    adapted3d = exposure.equalize_adapthist(img3d, kernel_size=5, clip_limit=0.05)
    assert img2d.shape == adapted2d.shape
    assert img3d.shape == adapted3d.shape
    assert np.mean(np.abs(adapted2d - adapted3d[adapted3d.shape[0] // 2])) < 0.02