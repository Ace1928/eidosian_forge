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
@pytest.mark.parametrize('normalization', [None, 'phase'])
def test_subpixel_precision(normalization):
    reference_image = fft.fftn(camera())
    subpixel_shift = (-2.4, 1.32)
    shifted_image = fourier_shift(reference_image, subpixel_shift)
    result, _, _ = phase_cross_correlation(reference_image, shifted_image, upsample_factor=100, space='fourier', normalization=normalization)
    assert_allclose(result[:2], -np.array(subpixel_shift), atol=0.05)