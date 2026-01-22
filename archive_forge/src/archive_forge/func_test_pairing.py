import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
@pytest.mark.parametrize('pairing, sos', [('nearest', np.array([[1.0, 1.0, 0.5, 1.0, -0.75, 0.0], [1.0, 1.0, 0.0, 1.0, -1.6, 0.65]])), ('keep_odd', np.array([[1.0, 1.0, 0, 1.0, -0.75, 0.0], [1.0, 1.0, 0.5, 1.0, -1.6, 0.65]])), ('minimal', np.array([[0.0, 1.0, 1.0, 0.0, 1.0, -0.75], [1.0, 1.0, 0.5, 1.0, -1.6, 0.65]]))])
def test_pairing(self, pairing, sos):
    z1 = np.array([-1, -0.5 - 0.5j, -0.5 + 0.5j])
    p1 = np.array([0.75, 0.8 + 0.1j, 0.8 - 0.1j])
    sos2 = zpk2sos(z1, p1, 1, pairing=pairing)
    assert_array_almost_equal(sos, sos2, decimal=4)