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
def test_binom_nooverflow_8346(self):
    dataset = [(1000, 500, 2.7028824094543655e+299), (1002, 501, 1.0800739688079122e+300), (1004, 502, 4.315992791690581e+300), (1006, 503, 1.7246810161626378e+301), (1008, 504, 6.891880092364192e+301), (1010, 505, 2.7540225794833545e+302), (1012, 506, 1.1005204853192376e+303), (1014, 507, 4.3977406375873285e+303), (1016, 508, 1.7573648610831252e+304), (1018, 509, 7.022554277884237e+304), (1020, 510, 2.8062677682996225e+305), (1022, 511, 1.1214087637706124e+306), (1024, 512, 4.481254552098971e+306), (1026, 513, 1.790754743041499e+307), (1028, 514, 7.156051054877897e+307)]
    dataset = np.asarray(dataset)
    FuncData(cephes.binom, dataset, (0, 1), 2, rtol=1e-12).check()