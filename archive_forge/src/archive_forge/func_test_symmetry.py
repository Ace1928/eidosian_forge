import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_symmetry(self):
    for N in np.arange(1, 26):
        for ftype in ('butter', 'bessel', 'cheby1', 'cheby2', 'ellip'):
            z, p, k = iirfilter(N, 1.1, 1, 20, 'low', analog=True, ftype=ftype, output='zpk')
            assert_array_equal(sorted(z), sorted(z.conj()))
            assert_array_equal(sorted(p), sorted(p.conj()))
            assert_equal(k, np.real(k))
            b, a = iirfilter(N, 1.1, 1, 20, 'low', analog=True, ftype=ftype, output='ba')
            assert_(issubclass(b.dtype.type, np.floating))
            assert_(issubclass(a.dtype.type, np.floating))