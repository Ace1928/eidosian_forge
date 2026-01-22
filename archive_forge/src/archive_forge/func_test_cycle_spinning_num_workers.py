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
def test_cycle_spinning_num_workers():
    img = astro_gray
    sigma = 0.1
    rstate = np.random.default_rng(1234)
    noisy = img.copy() + 0.1 * rstate.standard_normal(img.shape)
    denoise_func = restoration.denoise_wavelet
    func_kw = dict(sigma=sigma, channel_axis=-1, rescale_sigma=True)
    dn_cc1 = restoration.cycle_spin(noisy, denoise_func, max_shifts=1, func_kw=func_kw, channel_axis=None, num_workers=1)
    dn_cc1_ = restoration.cycle_spin(noisy, denoise_func, max_shifts=1, func_kw=func_kw, num_workers=1)
    assert_array_equal(dn_cc1, dn_cc1_)
    with expected_warnings([DASK_NOT_INSTALLED_WARNING]):
        dn_cc2 = restoration.cycle_spin(noisy, denoise_func, max_shifts=1, func_kw=func_kw, channel_axis=None, num_workers=4)
        dn_cc3 = restoration.cycle_spin(noisy, denoise_func, max_shifts=1, func_kw=func_kw, channel_axis=None, num_workers=None)
    assert_array_almost_equal(dn_cc1, dn_cc2)
    assert_array_almost_equal(dn_cc1, dn_cc3)