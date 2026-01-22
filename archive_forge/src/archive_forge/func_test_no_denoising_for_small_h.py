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
def test_no_denoising_for_small_h(fast_mode, dtype):
    img = np.zeros((40, 40))
    img[10:-10, 10:-10] = 1.0
    img += 0.3 * np.random.standard_normal(img.shape)
    img = img.astype(dtype)
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.01, fast_mode=fast_mode, channel_axis=None)
    assert np.allclose(denoised, img)
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.01, fast_mode=fast_mode, channel_axis=None)
    assert np.allclose(denoised, img)