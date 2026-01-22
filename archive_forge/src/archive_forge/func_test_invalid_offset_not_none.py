import math
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from skimage._shared.utils import _supported_float_type
from skimage.morphology.grayreconstruct import reconstruction
def test_invalid_offset_not_none():
    """Test reconstruction with invalid not None offset parameter"""
    image = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 3, 1], [1, 1, 1, 1, 1, 1, 1, 1]])
    mask = np.array([[4, 4, 4, 1, 1, 1, 1, 1], [4, 4, 4, 1, 1, 1, 1, 1], [4, 4, 4, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 4, 4, 4], [1, 1, 1, 1, 1, 4, 4, 4], [1, 1, 1, 1, 1, 4, 4, 4]])
    with pytest.raises(ValueError):
        reconstruction(image, mask, method='dilation', footprint=np.ones((3, 3)), offset=np.array([3, 0]))