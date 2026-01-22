import numpy as np
import pytest
from numpy.testing import (
from scipy.ndimage import fourier_shift, shift as real_shift
import scipy.fft as fft
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.data import camera, brain
from skimage.io import imread
from skimage.registration._masked_phase_cross_correlation import (
from skimage.registration import phase_cross_correlation
def test_cross_correlate_masked_over_axes():
    """Masked normalized cross-correlation over axes should be
    equivalent to a loop over non-transform axes."""
    np.random.seed(23)
    arr1 = np.random.random((8, 8, 5))
    arr2 = np.random.random((8, 8, 5))
    m1 = np.random.choice([True, False], arr1.shape)
    m2 = np.random.choice([True, False], arr2.shape)
    with_loop = np.empty_like(arr1, dtype=complex)
    for index in range(arr1.shape[-1]):
        with_loop[:, :, index] = cross_correlate_masked(arr1[:, :, index], arr2[:, :, index], m1[:, :, index], m2[:, :, index], axes=(0, 1), mode='same')
    over_axes = cross_correlate_masked(arr1, arr2, m1, m2, axes=(0, 1), mode='same')
    assert_array_almost_equal(with_loop, over_axes)