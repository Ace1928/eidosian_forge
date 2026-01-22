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
def test_covariances():
    """Covariance matrix"""
    x = ufloat(1, 0.1)
    y = -2 * x + 10
    z = -3 * x
    covs = uncert_core.covariance_matrix([x, y, z])
    assert numbers_close(covs[0][0], 0.01)
    assert numbers_close(covs[1][1], 0.04)
    assert numbers_close(covs[2][2], 0.09)
    assert numbers_close(covs[0][1], -0.02)