import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from skimage import data
from skimage import exposure
from skimage._shared.utils import _supported_float_type
from skimage.exposure import histogram_matching
@pytest.mark.parametrize('channel_axis', (0, 1, -1))
def test_match_histograms_channel_axis(self, channel_axis):
    """Assert that pdf of matched image is close to the reference's pdf for
        all channels and all values of matched"""
    image = np.moveaxis(self.image_rgb, -1, channel_axis)
    reference = np.moveaxis(self.template_rgb, -1, channel_axis)
    matched = exposure.match_histograms(image, reference, channel_axis=channel_axis)
    assert matched.dtype == image.dtype
    matched = np.moveaxis(matched, channel_axis, -1)
    reference = np.moveaxis(reference, channel_axis, -1)
    matched_pdf = self._calculate_image_empirical_pdf(matched)
    reference_pdf = self._calculate_image_empirical_pdf(reference)
    for channel in range(len(matched_pdf)):
        reference_values, reference_quantiles = reference_pdf[channel]
        matched_values, matched_quantiles = matched_pdf[channel]
        for i, matched_value in enumerate(matched_values):
            closest_id = np.abs(reference_values - matched_value).argmin()
            assert_almost_equal(matched_quantiles[i], reference_quantiles[closest_id], decimal=1)