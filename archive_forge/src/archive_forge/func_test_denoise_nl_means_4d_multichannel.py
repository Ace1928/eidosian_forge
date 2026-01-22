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
def test_denoise_nl_means_4d_multichannel():
    img = np.zeros((8, 8, 8, 4, 4))
    img[2:-2, 2:-2, 2:-2, 1:-1, :] = 1.0
    sigma = 0.3
    imgn = img + sigma * np.random.randn(*img.shape)
    psnr_noisy = peak_signal_noise_ratio(img, imgn, data_range=1.0)
    denoised_4dmc = restoration.denoise_nl_means(imgn, 3, 3, h=0.35 * sigma, fast_mode=True, channel_axis=-1, sigma=sigma)
    psnr_4dmc = peak_signal_noise_ratio(img, denoised_4dmc, data_range=1.0)
    assert psnr_4dmc > psnr_noisy