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
def norm_brightness_err(img1, img2):
    """Normalized Absolute Mean Brightness Error between two images

    Parameters
    ----------
    img1 : array-like
    img2 : array-like

    Returns
    -------
    norm_brightness_error : float
        Normalized absolute mean brightness error
    """
    if img1.ndim == 3:
        img1, img2 = (rgb2gray(img1), rgb2gray(img2))
    ambe = np.abs(img1.mean() - img2.mean())
    nbe = ambe / dtype_range[img1.dtype.type][1]
    return nbe