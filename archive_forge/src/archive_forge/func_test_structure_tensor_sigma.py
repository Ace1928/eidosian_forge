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
@pytest.mark.parametrize('ndim', [2, 3])
def test_structure_tensor_sigma(ndim):
    img = np.zeros((5,) * ndim)
    img[[2] * ndim] = 1
    A_default = structure_tensor(img, sigma=0.1, order='rc')
    A_tuple = structure_tensor(img, sigma=(0.1,) * ndim, order='rc')
    A_list = structure_tensor(img, sigma=[0.1] * ndim, order='rc')
    assert_array_equal(A_tuple, A_default)
    assert_array_equal(A_list, A_default)
    with pytest.raises(ValueError):
        structure_tensor(img, sigma=(0.1,) * (ndim - 1), order='rc')
    with pytest.raises(ValueError):
        structure_tensor(img, sigma=[0.1] * (ndim + 1), order='rc')