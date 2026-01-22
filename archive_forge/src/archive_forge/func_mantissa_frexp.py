from __future__ import division
from __future__ import absolute_import
import sys
import math
from uncertainties import ufloat
import uncertainties.core as uncert_core
import uncertainties.umath_core as umath_core
from . import test_uncertainties
def mantissa_frexp(x):
    return umath_core.frexp(x)[0]