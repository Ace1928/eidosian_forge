import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_fir_ba_output(self):
    b, _ = gammatone(15, 'fir', fs=1000)
    b2 = [0.0, 0.00022608075649884, 0.0015077903981357, 0.0042033687753998, 0.0081508962726503, 0.012890059089154, 0.017833890391666, 0.022392613558564, 0.026055195863104, 0.028435872863284, 0.029293319149544, 0.02852976858014, 0.026176557156294, 0.022371510270395, 0.017332485267759]
    assert_allclose(b, b2)