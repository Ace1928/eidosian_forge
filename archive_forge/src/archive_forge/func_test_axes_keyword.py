from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close, SCIPY_DEVICE
from scipy import fft
@skip_if_array_api_backend('torch')
@array_api_compatible
def test_axes_keyword(self, xp):
    freqs = xp.asarray([[0, 1, 2], [3, 4, -4], [-3, -2, -1]])
    shifted = xp.asarray([[-1, -3, -2], [2, 0, 1], [-4, 3, 4]])
    xp_assert_close(fft.fftshift(freqs, axes=(0, 1)), shifted)
    xp_assert_close(fft.fftshift(freqs, axes=0), fft.fftshift(freqs, axes=(0,)))
    xp_assert_close(fft.ifftshift(shifted, axes=(0, 1)), freqs)
    xp_assert_close(fft.ifftshift(shifted, axes=0), fft.ifftshift(shifted, axes=(0,)))
    xp_assert_close(fft.fftshift(freqs), shifted)
    xp_assert_close(fft.ifftshift(shifted), freqs)