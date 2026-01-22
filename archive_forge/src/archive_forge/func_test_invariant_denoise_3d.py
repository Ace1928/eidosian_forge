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
def test_invariant_denoise_3d():
    denoised_img_3d = denoise_invariant(noisy_img_3d, _denoise_wavelet)
    denoised_mse = mse(denoised_img_3d, test_img_3d)
    original_mse = mse(noisy_img_3d, test_img_3d)
    assert_(denoised_mse < original_mse)