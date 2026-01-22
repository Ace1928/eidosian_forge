import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from skimage import data
from skimage import exposure
from skimage._shared.utils import _supported_float_type
from skimage.exposure import histogram_matching
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_match_histograms_float_dtype(self, dtype):
    """float16 or float32 inputs give float32 output"""
    image = self.image_rgb.astype(dtype, copy=False)
    reference = self.template_rgb.astype(dtype, copy=False)
    matched = exposure.match_histograms(image, reference)
    assert matched.dtype == _supported_float_type(dtype)