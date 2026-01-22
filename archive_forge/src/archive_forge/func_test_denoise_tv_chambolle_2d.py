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
@pytest.mark.parametrize('dtype', float_dtypes)
def test_denoise_tv_chambolle_2d(dtype):
    img = astro_gray.astype(dtype, copy=True)
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)
    denoised_astro = restoration.denoise_tv_chambolle(img, weight=0.1)
    assert denoised_astro.dtype == _supported_float_type(img.dtype)
    from scipy import ndimage as ndi
    float_dtype = _supported_float_type(img.dtype)
    img = img.astype(float_dtype, copy=False)
    grad = ndi.morphological_gradient(img, size=(3, 3))
    grad_denoised = ndi.morphological_gradient(denoised_astro, size=(3, 3))
    assert grad_denoised.dtype == float_dtype
    assert np.sqrt((grad_denoised ** 2).sum()) < np.sqrt((grad ** 2).sum())