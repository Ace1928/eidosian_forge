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
def test_bi_zeros(self):
    bi = special.bi_zeros(2)
    bia = (array([-1.17371322, -3.271093]), array([-2.29443968, -4.07315509]), array([-0.45494438, 0.39652284]), array([0.60195789, -0.76031014]))
    assert_array_almost_equal(bi, bia, 4)
    bi = special.bi_zeros(5)
    assert_array_almost_equal(bi[0], array([-1.173713222709127, -3.271093302836352, -4.830737841662016, -6.169852128310251, -7.376762079367764]), 11)
    assert_array_almost_equal(bi[1], array([-2.294439682614122, -4.073155089071828, -5.512395729663599, -6.781294445990305, -7.940178689168587]), 10)
    assert_array_almost_equal(bi[2], array([-0.454944383639657, 0.396522836094465, -0.367969161486959, 0.349499116831805, -0.336026240133662]), 11)
    assert_array_almost_equal(bi[3], array([0.601957887976239, -0.760310141492801, 0.836991012619261, -0.88947990142654, 0.929983638568022]), 10)