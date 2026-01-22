import numpy as np
import pytest
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_equal
from skimage._shared._warnings import expected_warnings
def test_non_bool_out():
    image = test_holes_image.copy()
    expected_out = np.empty_like(image, dtype=int)
    with testing.raises(TypeError):
        remove_small_holes(image, area_threshold=3, out=expected_out)