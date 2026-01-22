from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import zip
from builtins import map
from builtins import range
import copy
import weakref
import math
from math import isnan, isinf
import random
import sys
import uncertainties.core as uncert_core
from uncertainties.core import ufloat, AffineScalarFunc, ufloat_fromstr
from uncertainties import umath
def test_ufloat_fromstr():
    """Input of numbers with uncertainties as a string"""
    tests = {'-1.23(3.4)': (-1.23, 3.4), '  -1.23(3.4)  ': (-1.23, 3.4), '-1.34(5)': (-1.34, 0.05), '1(6)': (1, 6), '3(4.2)': (3, 4.2), '-9(2)': (-9, 2), '1234567(1.2)': (1234567, 1.2), '12.345(15)': (12.345, 0.015), '-12.3456(78)e-6': (-1.23456e-05, 7.8e-09), '0.29': (0.29, 0.01), '31.': (31, 1), '-31.': (-31, 1), '31': (31, 1), '-3.1e10': (-31000000000.0, 1000000000.0), '169.0(7)': (169, 0.7), '-0.1+/-1': (-0.1, 1), '-13e-2+/-1e2': (-0.13, 100.0), '-14.(15)': (-14, 15), '-100.0(15)': (-100, 1.5), '14.(15)': (14, 15), '(3.141+/-0.001)E+02': (314.1, 0.1), u'(3.141±0.001)E+02': (314.1, 0.1), u'3.141E+02±0.001e2': (314.1, 0.1), u'(3.141 ± 0.001) × 10²': (314.1, 0.1), '(2 +/- 0.1)': (2, 0.1), u'(3.141±nan)E+02': (314.1, float('nan')), '3.141e+02+/-nan': (314.1, float('nan')), '3.4(nan)e10': (34000000000.0, float('nan')), 'nan+/-3.14e2': (float('nan'), 314), '(-3.1415 +/- 1e-4)e+200': (-3.1415e+200, 1e+196), '(-3.1415e-10 +/- 1e-4)e+200': (-3.1415e+190, 1e+196), '-3(0.)': (-3, 0)}
    for representation, values in tests.items():
        representation = u'  {}  '.format(representation)
        num = ufloat_fromstr(representation)
        assert numbers_close(num.nominal_value, values[0])
        assert numbers_close(num.std_dev, values[1])
        assert num.tag is None
        num = ufloat_fromstr(representation, 'test variable')
        assert numbers_close(num.nominal_value, values[0])
        assert numbers_close(num.std_dev, values[1])
        assert num.tag == 'test variable'
        num = ufloat_fromstr(representation, tag='test variable')
        assert numbers_close(num.nominal_value, values[0])
        assert numbers_close(num.std_dev, values[1])
        assert num.tag == 'test variable'
        num = ufloat(representation)
        assert numbers_close(num.nominal_value, values[0])
        assert numbers_close(num.std_dev, values[1])
        assert num.tag is None
        num = ufloat(representation, 'test variable')
        assert numbers_close(num.nominal_value, values[0])
        assert numbers_close(num.std_dev, values[1])
        assert num.tag == 'test variable'
        num = ufloat(representation, tag='test variable')
        assert numbers_close(num.nominal_value, values[0])
        assert numbers_close(num.std_dev, values[1])
        assert num.tag == 'test variable'