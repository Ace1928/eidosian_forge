import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import find_boundaries, mark_boundaries
def test_find_boundaries_bool():
    image = np.zeros((5, 5), dtype=bool)
    image[2:5, 2:5] = True
    ref = np.array([[False, False, False, False, False], [False, False, True, True, True], [False, True, True, True, True], [False, True, True, False, False], [False, True, True, False, False]], dtype=bool)
    result = find_boundaries(image)
    assert_array_equal(result, ref)