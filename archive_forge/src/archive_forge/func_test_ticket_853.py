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
def test_ticket_853(self):
    """Negative-order Bessels"""
    assert_allclose(special.jv(-1, 1), -0.4400505857449335)
    assert_allclose(special.jv(-2, 1), 0.1149034849319005)
    assert_allclose(special.yv(-1, 1), 0.7812128213002887)
    assert_allclose(special.yv(-2, 1), -1.650682606816255)
    assert_allclose(special.iv(-1, 1), 0.5651591039924851)
    assert_allclose(special.iv(-2, 1), 0.1357476697670383)
    assert_allclose(special.kv(-1, 1), 0.6019072301972347)
    assert_allclose(special.kv(-2, 1), 1.624838898635178)
    assert_allclose(special.jv(-0.5, 1), 0.4310988680183761)
    assert_allclose(special.yv(-0.5, 1), 0.6713967071418031)
    assert_allclose(special.iv(-0.5, 1), 1.231200214592967)
    assert_allclose(special.kv(-0.5, 1), 0.4610685044478945)
    assert_allclose(special.jv(-1, 1 + 0j), -0.4400505857449335)
    assert_allclose(special.jv(-2, 1 + 0j), 0.1149034849319005)
    assert_allclose(special.yv(-1, 1 + 0j), 0.7812128213002887)
    assert_allclose(special.yv(-2, 1 + 0j), -1.650682606816255)
    assert_allclose(special.iv(-1, 1 + 0j), 0.5651591039924851)
    assert_allclose(special.iv(-2, 1 + 0j), 0.1357476697670383)
    assert_allclose(special.kv(-1, 1 + 0j), 0.6019072301972347)
    assert_allclose(special.kv(-2, 1 + 0j), 1.624838898635178)
    assert_allclose(special.jv(-0.5, 1 + 0j), 0.4310988680183761)
    assert_allclose(special.jv(-0.5, 1 + 1j), 0.2628946385649065 - 0.827050182040562j)
    assert_allclose(special.yv(-0.5, 1 + 0j), 0.6713967071418031)
    assert_allclose(special.yv(-0.5, 1 + 1j), 0.967901282890131 + 0.0602046062142816j)
    assert_allclose(special.iv(-0.5, 1 + 0j), 1.231200214592967)
    assert_allclose(special.iv(-0.5, 1 + 1j), 0.77070737376928 + 0.39891821043561j)
    assert_allclose(special.kv(-0.5, 1 + 0j), 0.4610685044478945)
    assert_allclose(special.kv(-0.5, 1 + 1j), 0.06868578341999 - 0.38157825981268j)
    assert_allclose(special.jve(-0.5, 1 + 0.3j), special.jv(-0.5, 1 + 0.3j) * exp(-0.3))
    assert_allclose(special.yve(-0.5, 1 + 0.3j), special.yv(-0.5, 1 + 0.3j) * exp(-0.3))
    assert_allclose(special.ive(-0.5, 0.3 + 1j), special.iv(-0.5, 0.3 + 1j) * exp(-0.3))
    assert_allclose(special.kve(-0.5, 0.3 + 1j), special.kv(-0.5, 0.3 + 1j) * exp(0.3 + 1j))
    assert_allclose(special.hankel1(-0.5, 1 + 1j), special.jv(-0.5, 1 + 1j) + 1j * special.yv(-0.5, 1 + 1j))
    assert_allclose(special.hankel2(-0.5, 1 + 1j), special.jv(-0.5, 1 + 1j) - 1j * special.yv(-0.5, 1 + 1j))