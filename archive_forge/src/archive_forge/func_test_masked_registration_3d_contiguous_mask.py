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
def test_masked_registration_3d_contiguous_mask():
    """masked_register_translation should be able to register translations
    between volumes with contiguous masks."""
    ref_vol = brain()[:, ::2, ::2]
    offset = (1, -5, 10)
    ref_mask = np.zeros_like(ref_vol, dtype=bool)
    ref_mask[:-2, 75:100, 75:100] = True
    ref_shifted = real_shift(ref_vol, offset)
    measured_offset = masked_register_translation(ref_vol, ref_shifted, reference_mask=ref_mask, moving_mask=ref_mask)
    assert_equal(offset, -np.array(measured_offset))