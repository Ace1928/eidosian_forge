import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from skimage import data
from skimage import exposure
from skimage._shared.utils import _supported_float_type
from skimage.exposure import histogram_matching
def test_match_histograms_consistency(self):
    """ensure equivalent results for float and integer-based code paths"""
    image_u8 = self.image_rgb
    reference_u8 = self.template_rgb
    image_f64 = self.image_rgb.astype(np.float64)
    reference_f64 = self.template_rgb.astype(np.float64, copy=False)
    matched_u8 = exposure.match_histograms(image_u8, reference_u8)
    matched_f64 = exposure.match_histograms(image_f64, reference_f64)
    assert_array_almost_equal(matched_u8.astype(np.float64), matched_f64)