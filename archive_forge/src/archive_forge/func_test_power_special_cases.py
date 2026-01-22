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
def test_power_special_cases():
    """
    Checks special cases of x**p.
    """
    power_special_cases(pow)
    positive = ufloat(0.3, 0.01)
    negative = ufloat(-0.3, 0.01)
    try:
        pow(ufloat(0, 0), negative)
    except ZeroDivisionError:
        pass
    else:
        raise Exception('A proper exception should have been raised')
    try:
        pow(ufloat(0, 0.1), negative)
    except ZeroDivisionError:
        pass
    else:
        raise Exception('A proper exception should have been raised')
    try:
        result = pow(negative, positive)
    except ValueError:
        pass
    else:
        raise Exception('A proper exception should have been raised')