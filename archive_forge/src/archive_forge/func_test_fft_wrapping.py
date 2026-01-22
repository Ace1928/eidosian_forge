import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_fft_wrapping(self):
    bs = list()
    as_ = list()
    hs_whole = list()
    hs_half = list()
    t = np.linspace(0, 1, 3, endpoint=False)
    bs.append(np.sin(2 * np.pi * t))
    as_.append(3.0)
    hs_whole.append([0, -0.5j, 0.5j])
    hs_half.append([0, np.sqrt(1.0 / 12.0), -0.5j])
    t = np.linspace(0, 1, 4, endpoint=False)
    bs.append(np.sin(2 * np.pi * t))
    as_.append(0.5)
    hs_whole.append([0, -4j, 0, 4j])
    hs_half.append([0, np.sqrt(8), -4j, -np.sqrt(8)])
    del t
    for ii, b in enumerate(bs):
        a = as_[ii]
        expected_w = np.linspace(0, 2 * np.pi, len(b), endpoint=False)
        w, h = freqz(b, a, worN=expected_w, whole=True)
        err_msg = f'b = {b}, a={a}'
        assert_array_almost_equal(w, expected_w, err_msg=err_msg)
        assert_array_almost_equal(h, hs_whole[ii], err_msg=err_msg)
        w, h = freqz(b, a, worN=len(b), whole=True)
        assert_array_almost_equal(w, expected_w, err_msg=err_msg)
        assert_array_almost_equal(h, hs_whole[ii], err_msg=err_msg)
        expected_w = np.linspace(0, np.pi, len(b), endpoint=False)
        w, h = freqz(b, a, worN=expected_w, whole=False)
        assert_array_almost_equal(w, expected_w, err_msg=err_msg)
        assert_array_almost_equal(h, hs_half[ii], err_msg=err_msg)
        w, h = freqz(b, a, worN=len(b), whole=False)
        assert_array_almost_equal(w, expected_w, err_msg=err_msg)
        assert_array_almost_equal(h, hs_half[ii], err_msg=err_msg)
    rng = np.random.RandomState(0)
    for ii in range(2, 10):
        b = rng.randn(ii)
        for kk in range(2):
            a = rng.randn(1) if kk == 0 else rng.randn(3)
            for jj in range(2):
                if jj == 1:
                    b = b + rng.randn(ii) * 1j
                expected_w = np.linspace(0, 2 * np.pi, ii, endpoint=False)
                w, expected_h = freqz(b, a, worN=expected_w, whole=True)
                assert_array_almost_equal(w, expected_w)
                w, h = freqz(b, a, worN=ii, whole=True)
                assert_array_almost_equal(w, expected_w)
                assert_array_almost_equal(h, expected_h)
                expected_w = np.linspace(0, np.pi, ii, endpoint=False)
                w, expected_h = freqz(b, a, worN=expected_w, whole=False)
                assert_array_almost_equal(w, expected_w)
                w, h = freqz(b, a, worN=ii, whole=False)
                assert_array_almost_equal(w, expected_w)
                assert_array_almost_equal(h, expected_h)