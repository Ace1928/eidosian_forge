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
@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_denoise_nl_means_3d(fast_mode, dtype):
    img = np.zeros((12, 12, 8), dtype=dtype)
    img[5:-5, 5:-5, 2:-2] = 1.0
    sigma = 0.3
    imgn = img + sigma * np.random.standard_normal(img.shape)
    imgn = imgn.astype(dtype)
    psnr_noisy = peak_signal_noise_ratio(img, imgn)
    for s in [sigma, 0]:
        denoised = restoration.denoise_nl_means(imgn, 3, 4, h=0.75 * sigma, fast_mode=fast_mode, channel_axis=None, sigma=s)
        assert peak_signal_noise_ratio(img, denoised) > psnr_noisy