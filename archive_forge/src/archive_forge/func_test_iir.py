import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_iir(self):
    b, a = butter(4, 0.1)
    w = np.linspace(0, pi, num=10, endpoint=False)
    w, gd = group_delay((b, a), w=w)
    matlab_gd = np.array([8.249313898506037, 11.958947880907104, 2.452325615326005, 1.048918665702008, 0.611382575635897, 0.418293269460578, 0.317932917836572, 0.261371844762525, 0.229038045801298, 0.212185774208521])
    assert_array_almost_equal(gd, matlab_gd)