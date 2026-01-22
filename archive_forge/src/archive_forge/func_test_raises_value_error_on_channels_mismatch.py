import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from skimage import data
from skimage import exposure
from skimage._shared.utils import _supported_float_type
from skimage.exposure import histogram_matching
@pytest.mark.parametrize('image, reference', [(image_rgb, template_rgb[:, :, 0]), (image_rgb[:, :, 0], template_rgb)])
def test_raises_value_error_on_channels_mismatch(self, image, reference):
    with pytest.raises(ValueError):
        exposure.match_histograms(image, reference)