import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_bad_pass_zero(self):
    """Test degenerate pass_zero cases."""
    with assert_raises(ValueError, match='pass_zero must be'):
        firwin(41, 0.5, pass_zero='foo')
    with assert_raises(TypeError, match='cannot be interpreted'):
        firwin(41, 0.5, pass_zero=1.0)
    for pass_zero in ('lowpass', 'highpass'):
        with assert_raises(ValueError, match='cutoff must have one'):
            firwin(41, [0.5, 0.6], pass_zero=pass_zero)
    for pass_zero in ('bandpass', 'bandstop'):
        with assert_raises(ValueError, match='must have at least two'):
            firwin(41, [0.5], pass_zero=pass_zero)