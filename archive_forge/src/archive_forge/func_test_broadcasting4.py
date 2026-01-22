import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_broadcasting4(self):
    np.random.seed(123)
    b = np.random.rand(4, 2, 1, 1)
    a = np.random.rand(5, 2, 1, 1)
    for whole in [False, True]:
        for worN in [np.random.rand(6, 7), np.empty((6, 0))]:
            w, h = freqz(b, a, worN=worN, whole=whole)
            assert_allclose(w, worN, rtol=1e-14)
            assert_equal(h.shape, (2,) + worN.shape)
            for k in range(2):
                ww, hh = freqz(b[:, k, 0, 0], a[:, k, 0, 0], worN=worN.ravel(), whole=whole)
                assert_allclose(ww, worN.ravel(), rtol=1e-14)
                assert_allclose(hh, h[k, :, :].ravel())