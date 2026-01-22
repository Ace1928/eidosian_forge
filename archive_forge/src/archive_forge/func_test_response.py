import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_response(self):
    N = 51
    f = 0.5
    h = firwin(N, f)
    self.check_response(h, [(0.25, 1), (0.75, 0)])
    h = firwin(N + 1, f, window='nuttall')
    self.check_response(h, [(0.25, 1), (0.75, 0)])
    h = firwin(N + 2, f, pass_zero=False)
    self.check_response(h, [(0.25, 0), (0.75, 1)])
    f1, f2, f3, f4 = (0.2, 0.4, 0.6, 0.8)
    h = firwin(N + 3, [f1, f2], pass_zero=False)
    self.check_response(h, [(0.1, 0), (0.3, 1), (0.5, 0)])
    h = firwin(N + 4, [f1, f2])
    self.check_response(h, [(0.1, 1), (0.3, 0), (0.5, 1)])
    h = firwin(N + 5, [f1, f2, f3, f4], pass_zero=False, scale=False)
    self.check_response(h, [(0.1, 0), (0.3, 1), (0.5, 0), (0.7, 1), (0.9, 0)])
    h = firwin(N + 6, [f1, f2, f3, f4])
    self.check_response(h, [(0.1, 1), (0.3, 0), (0.5, 1), (0.7, 0), (0.9, 1)])
    h = firwin(N + 7, 0.1, width=0.03)
    self.check_response(h, [(0.05, 1), (0.75, 0)])
    h = firwin(N + 8, 0.1, pass_zero=False)
    self.check_response(h, [(0.05, 0), (0.75, 1)])