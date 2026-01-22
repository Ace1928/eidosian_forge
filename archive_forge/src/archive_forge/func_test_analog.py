import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_analog(self):
    wp = [1000, 6000]
    ws = [2000, 5000]
    rp = 3
    rs = 90
    N, Wn = ellipord(wp, ws, rp, rs, True)
    b, a = ellip(N, rp, rs, Wn, 'bs', True)
    w, h = freqs(b, a)
    assert_array_less(-rp - 0.1, dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
    assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]), -rs + 0.1)
    assert_equal(N, 8)
    assert_allclose(Wn, [1666.6666, 6000])
    assert_equal(ellipord(1, 1.2, 1, 80, analog=True)[0], 9)