import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data, draw, img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.feature import (
from skimage.morphology import cube, octagon
def test_blank_image_nans():
    """Some of the corner detectors had a weakness in terms of returning
    NaN when presented with regions of constant intensity. This should
    be fixed by now. We test whether each detector returns something
    finite in the case of constant input"""
    detectors = [corner_moravec, corner_harris, corner_shi_tomasi, corner_kitchen_rosenfeld, corner_foerstner]
    constant_image = np.zeros((20, 20))
    for det in detectors:
        response = det(constant_image)
        assert np.all(np.isfinite(response))