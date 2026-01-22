import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_fewer_zeros(self):
    """Test not the expected number of p/z (effectively at origin)."""
    sos = butter(3, 0.1, output='sos')
    z, p, k = sos2zpk(sos)
    assert len(z) == 4
    assert len(p) == 4
    sos = butter(12, [5.0, 30.0], 'bandpass', fs=1200.0, analog=False, output='sos')
    with pytest.warns(BadCoefficients, match='Badly conditioned'):
        z, p, k = sos2zpk(sos)
    assert len(z) == 24
    assert len(p) == 24