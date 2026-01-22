import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_rank_deficient(self):
    x = firls(21, [0, 0.1, 0.9, 1], [1, 1, 0, 0])
    w, h = freqz(x, fs=2.0)
    assert_allclose(np.abs(h[:2]), 1.0, atol=1e-05)
    assert_allclose(np.abs(h[-2:]), 0.0, atol=1e-06)
    x = firls(101, [0, 0.01, 0.99, 1], [1, 1, 0, 0])
    w, h = freqz(x, fs=2.0)
    mask = w < 0.01
    assert mask.sum() > 3
    assert_allclose(np.abs(h[mask]), 1.0, atol=0.0001)
    mask = w > 0.99
    assert mask.sum() > 3
    assert_allclose(np.abs(h[mask]), 0.0, atol=0.0001)