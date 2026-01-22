import numpy as np
from skimage.restoration import unwrap_phase
import sys
from skimage._shared import testing
from skimage._shared.testing import (
from skimage._shared._warnings import expected_warnings
def assert_phase_almost_equal(a, b, *args, **kwargs):
    """An assert_almost_equal insensitive to phase shifts of n*2*pi."""
    shift = 2 * np.pi * np.round((b.mean() - a.mean()) / (2 * np.pi))
    with expected_warnings(['invalid value encountered|\\A\\Z', 'divide by zero encountered|\\A\\Z']):
        print('assert_phase_allclose, abs', np.max(np.abs(a - (b - shift))))
        print('assert_phase_allclose, rel', np.max(np.abs((a - (b - shift)) / a)))
    if np.ma.isMaskedArray(a):
        assert_(np.ma.isMaskedArray(b))
        assert_array_equal(a.mask, b.mask)
        assert_(a.fill_value == b.fill_value)
        au = np.asarray(a)
        bu = np.asarray(b)
        with expected_warnings(['invalid value encountered|\\A\\Z', 'divide by zero encountered|\\A\\Z']):
            print('assert_phase_allclose, no mask, abs', np.max(np.abs(au - (bu - shift))))
            print('assert_phase_allclose, no mask, rel', np.max(np.abs((au - (bu - shift)) / au)))
    assert_array_almost_equal_nulp(a + shift, b, *args, **kwargs)