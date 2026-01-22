import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_integer_bw(self):
    float_result = waveforms.gausspulse('cutoff', bw=1.0)
    int_result = waveforms.gausspulse('cutoff', bw=1)
    err_msg = "Integer input 'bw=1' gives wrong result"
    assert_equal(int_result, float_result, err_msg=err_msg)