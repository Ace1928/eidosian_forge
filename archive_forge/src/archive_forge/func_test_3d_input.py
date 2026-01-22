import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import fourier_shift
import scipy.fft as fft
from skimage import img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import assert_stacklevel
from skimage._shared.utils import _supported_float_type
from skimage.data import camera, binary_blobs, eagle
from skimage.registration._phase_cross_correlation import (
def test_3d_input():
    phantom = img_as_float(binary_blobs(length=32, n_dim=3))
    reference_image = fft.fftn(phantom)
    shift = (-2.0, 1.0, 5.0)
    shifted_image = fourier_shift(reference_image, shift)
    result, error, diffphase = phase_cross_correlation(reference_image, shifted_image, space='fourier')
    assert_allclose(result, -np.array(shift), atol=0.05)
    subpixel_shift = (-2.3, 1.7, 5.4)
    shifted_image = fourier_shift(reference_image, subpixel_shift)
    result, error, diffphase = phase_cross_correlation(reference_image, shifted_image, upsample_factor=100, space='fourier')
    assert_allclose(result, -np.array(subpixel_shift), atol=0.05)