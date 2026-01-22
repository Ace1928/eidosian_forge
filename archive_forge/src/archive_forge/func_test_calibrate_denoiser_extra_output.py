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
def test_calibrate_denoiser_extra_output():
    parameter_ranges = {'sigma': np.linspace(0.1, 1, 5) / 2}
    _, (parameters_tested, losses) = calibrate_denoiser(noisy_img, _denoise_wavelet, denoise_parameters=parameter_ranges, extra_output=True)
    all_denoised = [denoise_invariant(noisy_img, _denoise_wavelet, denoiser_kwargs=denoiser_kwargs) for denoiser_kwargs in parameters_tested]
    ground_truth_losses = [mse(img, test_img) for img in all_denoised]
    assert_(np.argmin(losses) == np.argmin(ground_truth_losses))