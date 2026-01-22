import functools
import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_warns
from skimage import color, data, img_as_float, restoration
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration._denoise import _wavelet_threshold
@xfail_without_pywt
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_estimate_sigma_color(channel_axis):
    rstate = np.random.default_rng(1234)
    img = astro.copy()
    sigma = 0.1
    img += sigma * rstate.standard_normal(img.shape)
    img = np.moveaxis(img, -1, channel_axis)
    sigma_est = restoration.estimate_sigma(img, channel_axis=channel_axis, average_sigmas=True)
    assert_array_almost_equal(sigma, sigma_est, decimal=2)
    sigma_list = restoration.estimate_sigma(img, channel_axis=channel_axis, average_sigmas=False)
    assert_array_equal(len(sigma_list), img.shape[channel_axis])
    assert_array_almost_equal(sigma_list[0], sigma_est, decimal=2)
    if channel_axis % img.ndim == 2:
        assert_warns(UserWarning, restoration.estimate_sigma, img)