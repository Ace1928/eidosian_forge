import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import correlate
from skimage import draw
from skimage._shared.testing import fetch
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts
def test_00_01_zeros_masked(self):
    """Test skeletonize on an array that is completely masked"""
    result = medial_axis(np.zeros((10, 10), bool), np.zeros((10, 10), bool))
    assert np.all(result == False)