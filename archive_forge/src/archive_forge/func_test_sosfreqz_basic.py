import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_sosfreqz_basic(self):
    N = 500
    b, a = butter(4, 0.2)
    sos = butter(4, 0.2, output='sos')
    w, h = freqz(b, a, worN=N)
    w2, h2 = sosfreqz(sos, worN=N)
    assert_equal(w2, w)
    assert_allclose(h2, h, rtol=1e-10, atol=1e-14)
    b, a = ellip(3, 1, 30, (0.2, 0.3), btype='bandpass')
    sos = ellip(3, 1, 30, (0.2, 0.3), btype='bandpass', output='sos')
    w, h = freqz(b, a, worN=N)
    w2, h2 = sosfreqz(sos, worN=N)
    assert_equal(w2, w)
    assert_allclose(h2, h, rtol=1e-10, atol=1e-14)
    assert_raises(ValueError, sosfreqz, sos[:0])