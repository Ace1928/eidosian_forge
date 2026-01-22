import functools
import numpy as np
import pytest
from skimage._shared.testing import assert_
from skimage._shared.utils import _supported_float_type
from skimage.data import binary_blobs
from skimage.data import camera, chelsea
from skimage.metrics import mean_squared_error as mse
from skimage.restoration import calibrate_denoiser, denoise_wavelet
from skimage.restoration.j_invariant import denoise_invariant
from skimage.util import img_as_float, random_noise
from skimage.restoration.tests.test_denoise import xfail_without_pywt
@xfail_without_pywt
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_invariant_denoise_color(dtype):
    denoised_img_color = denoise_invariant(noisy_img_color.astype(dtype), _denoise_wavelet, denoiser_kwargs=dict(channel_axis=-1))
    denoised_mse = mse(denoised_img_color, test_img_color)
    original_mse = mse(noisy_img_color, test_img_color)
    assert denoised_mse < original_mse
    assert denoised_img_color.dtype == _supported_float_type(dtype)