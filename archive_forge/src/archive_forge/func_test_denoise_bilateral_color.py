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
@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_denoise_bilateral_color(channel_axis):
    img = checkerboard.copy()[:50, :50]
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)
    img = np.moveaxis(img, -1, channel_axis)
    out1 = restoration.denoise_bilateral(img, sigma_color=0.1, sigma_spatial=10, channel_axis=channel_axis)
    out2 = restoration.denoise_bilateral(img, sigma_color=0.2, sigma_spatial=20, channel_axis=channel_axis)
    img = np.moveaxis(img, channel_axis, -1)
    out1 = np.moveaxis(out1, channel_axis, -1)
    out2 = np.moveaxis(out2, channel_axis, -1)
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()