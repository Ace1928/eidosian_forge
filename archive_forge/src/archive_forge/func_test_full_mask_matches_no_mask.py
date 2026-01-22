import unittest
import numpy as np
import pytest
from skimage._shared.testing import assert_equal
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data, feature
from skimage.util import img_as_float
def test_full_mask_matches_no_mask(self):
    """The masked and unmasked algorithms should return the same result."""
    image = data.camera()
    for mode in ('constant', 'nearest', 'reflect'):
        assert_equal(feature.canny(image, mode=mode), feature.canny(image, mode=mode, mask=np.ones_like(image, dtype=bool)))