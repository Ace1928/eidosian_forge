from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close, SCIPY_DEVICE
from scipy import fft
@skip_if_array_api_gpu
@array_api_compatible
def test_xp_1d_defaults(self, xp):
    x = xp.arange(0, 1, 0.1)
    shape = None
    axes = None
    shape_expected = (10,)
    axes_expected = [0]
    shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
    assert shape_res == shape_expected
    assert axes_res == axes_expected