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
def test_rotated_img():
    """
    The harris filter should yield the same results with an image and it's
    rotation.
    """
    im = img_as_float(data.astronaut().mean(axis=2))
    im_rotated = im.T
    results = np.nonzero(corner_moravec(im))
    results_rotated = np.nonzero(corner_moravec(im_rotated))
    assert (np.sort(results[0]) == np.sort(results_rotated[1])).all()
    assert (np.sort(results[1]) == np.sort(results_rotated[0])).all()
    results = np.nonzero(corner_harris(im))
    results_rotated = np.nonzero(corner_harris(im_rotated))
    assert (np.sort(results[0]) == np.sort(results_rotated[1])).all()
    assert (np.sort(results[1]) == np.sort(results_rotated[0])).all()
    results = np.nonzero(corner_shi_tomasi(im))
    results_rotated = np.nonzero(corner_shi_tomasi(im_rotated))
    assert (np.sort(results[0]) == np.sort(results_rotated[1])).all()
    assert (np.sort(results[1]) == np.sort(results_rotated[0])).all()