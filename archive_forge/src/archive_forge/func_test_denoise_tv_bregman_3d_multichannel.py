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
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_denoise_tv_bregman_3d_multichannel(channel_axis):
    img_astro = astro.copy()
    denoised0 = restoration.denoise_tv_bregman(img_astro[..., 0], weight=60.0)
    img_astro = np.moveaxis(img_astro, -1, channel_axis)
    denoised = restoration.denoise_tv_bregman(img_astro, weight=60.0, channel_axis=channel_axis)
    _at = functools.partial(slice_at_axis, axis=channel_axis % img_astro.ndim)
    assert_array_equal(denoised0, denoised[_at(0)])