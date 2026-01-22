import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_sosfreqz_design_ellip(self):
    N, Wn = ellipord(0.3, 0.1, 3, 60)
    sos = ellip(N, 0.3, 60, Wn, 'high', output='sos')
    w, h = sosfreqz(sos)
    h = np.abs(h)
    w /= np.pi
    assert_allclose(20 * np.log10(h[w >= 0.3]), 0, atol=3.01)
    assert_allclose(h[w <= 0.1], 0.0, atol=0.0015)
    N, Wn = ellipord(0.3, 0.2, 0.5, 150)
    sos = ellip(N, 0.5, 150, Wn, 'high', output='sos')
    w, h = sosfreqz(sos)
    dB = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    w /= np.pi
    assert_allclose(dB[w >= 0.3], 0, atol=0.55)
    assert dB[w <= 0.2].max() < -150 * (1 - 1e-12)