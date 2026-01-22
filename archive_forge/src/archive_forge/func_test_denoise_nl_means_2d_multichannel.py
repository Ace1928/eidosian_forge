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
@pytest.mark.parametrize('fast_mode', [False, True])
@pytest.mark.parametrize('n_channels', [2, 3, 6])
@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_denoise_nl_means_2d_multichannel(fast_mode, n_channels, dtype):
    img = np.copy(astro[:50, :50])
    img = np.concatenate((img,) * 2)
    img = img.astype(dtype)
    sigma = 0.1
    imgn = img + sigma * np.random.standard_normal(img.shape)
    imgn = np.clip(imgn, 0, 1)
    imgn = imgn.astype(dtype)
    for s in [sigma, 0]:
        psnr_noisy = peak_signal_noise_ratio(img[..., :n_channels], imgn[..., :n_channels])
        denoised = restoration.denoise_nl_means(imgn[..., :n_channels], 3, 5, h=0.75 * sigma, fast_mode=fast_mode, channel_axis=-1, sigma=s)
        psnr_denoised = peak_signal_noise_ratio(denoised[..., :n_channels], img[..., :n_channels])
        assert psnr_denoised > psnr_noisy