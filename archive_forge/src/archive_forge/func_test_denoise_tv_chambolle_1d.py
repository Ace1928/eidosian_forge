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
def test_denoise_tv_chambolle_1d():
    """Apply the TV denoising algorithm on a 1D sinusoid."""
    x = 125 + 100 * np.sin(np.linspace(0, 8 * np.pi, 1000))
    x += 20 * np.random.rand(x.size)
    x = np.clip(x, 0, 255)
    res = restoration.denoise_tv_chambolle(x.astype(np.uint8), weight=0.1)
    assert res.dtype == float
    assert res.std() * 255 < x.std()