import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_broadcasting3(self):
    np.random.seed(123)
    N = 16
    b = np.random.rand(3, N)
    for whole in [False, True]:
        for worN in [N, np.linspace(0, 1, N)]:
            w, h = freqz(b, worN=worN, whole=whole)
            assert_equal(w.size, N)
            for k in range(N):
                bk = b[:, k]
                ww, hh = freqz(bk, worN=w[k], whole=whole)
                assert_allclose(ww, w[k])
                assert_allclose(hh, h[k])