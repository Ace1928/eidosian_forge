import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import correlate
from skimage import draw
from skimage._shared.testing import fetch
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts
def test_lut_generation(self):
    g123, g123p = _generate_thin_luts()
    assert_array_equal(g123, G123_LUT)
    assert_array_equal(g123p, G123P_LUT)