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
def test_denoise_bilateral_nan():
    img = np.full((50, 50), np.nan)
    with expected_warnings(['invalid|\\A\\Z']):
        out = restoration.denoise_bilateral(img, channel_axis=None)
    assert_array_equal(img, out)