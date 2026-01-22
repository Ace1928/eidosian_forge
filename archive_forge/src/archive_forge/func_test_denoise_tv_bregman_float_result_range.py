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
def test_denoise_tv_bregman_float_result_range():
    img = astro_gray.copy()
    int_astro = np.multiply(img, 255).astype(np.uint8)
    assert np.max(int_astro) > 1
    denoised_int_astro = restoration.denoise_tv_bregman(int_astro, weight=60.0)
    assert denoised_int_astro.dtype == float
    assert np.max(denoised_int_astro) <= 1.0
    assert np.min(denoised_int_astro) >= 0.0