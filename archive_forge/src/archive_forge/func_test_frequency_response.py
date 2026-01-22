import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_frequency_response(self):
    fs = 16000
    ftypes = ['fir', 'iir']
    for ftype in ftypes:
        b, a = gammatone(1000, ftype, fs=fs)
        freqs, response = freqz(b, a)
        response_max = np.max(np.abs(response))
        freq_hz = freqs[np.argmax(np.abs(response))] / (2 * np.pi / fs)
        assert_allclose(response_max, 1, rtol=0.01)
        assert_allclose(freq_hz, 1000, rtol=0.01)