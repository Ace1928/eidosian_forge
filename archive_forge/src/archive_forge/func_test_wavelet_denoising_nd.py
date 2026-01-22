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
@pytest.mark.parametrize('rescale_sigma, method, ndim', itertools.product([True, False], ['VisuShrink', 'BayesShrink'], range(1, 5)))
def test_wavelet_denoising_nd(rescale_sigma, method, ndim):
    rstate = np.random.default_rng(1234)
    if ndim < 3:
        img = 0.2 * np.ones((128,) * ndim)
    else:
        img = 0.2 * np.ones((16,) * ndim)
    img[(slice(5, 13),) * ndim] = 0.8
    sigma = 0.1
    noisy = img + sigma * rstate.standard_normal(img.shape)
    noisy = np.clip(noisy, 0, 1)
    denoised = restoration.denoise_wavelet(noisy, method=method, rescale_sigma=rescale_sigma)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    assert psnr_denoised > psnr_noisy