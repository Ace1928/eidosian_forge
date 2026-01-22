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
def test_estimate_sigma_gray():
    rstate = np.random.default_rng(1234)
    img = astro_gray.copy()
    sigma = 0.1
    img += sigma * rstate.standard_normal(img.shape)
    sigma_est = restoration.estimate_sigma(img, channel_axis=None)
    assert_array_almost_equal(sigma, sigma_est, decimal=2)