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
def test_uneven_dims(self, xp):
    """ Test 2D input, which has uneven dimension sizes """
    freqs = xp.asarray([[0, 1], [2, 3], [4, 5]])
    shift_dim0 = xp.asarray([[4, 5], [0, 1], [2, 3]])
    xp_assert_close(fft.fftshift(freqs, axes=0), shift_dim0)
    xp_assert_close(fft.ifftshift(shift_dim0, axes=0), freqs)
    xp_assert_close(fft.fftshift(freqs, axes=(0,)), shift_dim0)
    xp_assert_close(fft.ifftshift(shift_dim0, axes=[0]), freqs)
    shift_dim1 = xp.asarray([[1, 0], [3, 2], [5, 4]])
    xp_assert_close(fft.fftshift(freqs, axes=1), shift_dim1)
    xp_assert_close(fft.ifftshift(shift_dim1, axes=1), freqs)
    shift_dim_both = xp.asarray([[5, 4], [1, 0], [3, 2]])
    xp_assert_close(fft.fftshift(freqs, axes=(0, 1)), shift_dim_both)
    xp_assert_close(fft.ifftshift(shift_dim_both, axes=(0, 1)), freqs)
    xp_assert_close(fft.fftshift(freqs, axes=[0, 1]), shift_dim_both)
    xp_assert_close(fft.ifftshift(shift_dim_both, axes=[0, 1]), freqs)
    xp_assert_close(fft.fftshift(freqs, axes=None), shift_dim_both)
    xp_assert_close(fft.ifftshift(shift_dim_both, axes=None), freqs)
    xp_assert_close(fft.fftshift(freqs), shift_dim_both)
    xp_assert_close(fft.ifftshift(shift_dim_both), freqs)