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
def test_denoise_nl_means_wrong_dimension():
    img = np.zeros((5,))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=None)
    img = np.zeros((5, 3))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=-1)
    img = np.zeros((5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=-1, fast_mode=False)
    img = np.zeros((5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=None, fast_mode=False)
    img = np.zeros((5, 5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=-1, fast_mode=False)
    img = np.zeros((5, 5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=None)