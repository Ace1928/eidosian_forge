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
def test_int_div():
    """Integer division"""
    x = ufloat(3.9, 2) // 2
    assert x.nominal_value == 1.0
    assert x.std_dev == 0.0