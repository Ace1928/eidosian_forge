import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
def test_bi_zeros_big(self):
    z, zp, bi_zpx, bip_zx = special.bi_zeros(50000)
    _, _, bi_z, bip_z = special.airy(z)
    _, _, bi_zp, bip_zp = special.airy(zp)
    bi_envelope = 1 / abs(z) ** (1.0 / 4)
    bip_envelope = abs(zp) ** (1.0 / 4)
    assert_allclose(bi_zpx, bi_zp, rtol=1e-10)
    assert_allclose(bip_zx, bip_z, rtol=1e-10)
    assert_allclose(bi_z / bi_envelope, 0, atol=1e-10, rtol=0)
    assert_allclose(bip_zp / bip_envelope, 0, atol=1e-10, rtol=0)
    assert_allclose(z[:6], [-1.1737132227, -3.2710933028, -4.8307378417, -6.1698521283, -7.3767620794, -8.4919488465], rtol=1e-10)
    assert_allclose(zp[:6], [-2.2944396826, -4.0731550891, -5.5123957297, -6.781294446, -7.9401786892, -9.0195833588], rtol=1e-10)