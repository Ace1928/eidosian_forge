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
def test_basic_access_to_data():
    """Access to data from Variable and AffineScalarFunc objects."""
    x = ufloat(3.14, 0.01, 'x var')
    assert x.tag == 'x var'
    assert x.nominal_value == 3.14
    assert x.std_dev == 0.01
    y = x + 0
    assert type(y) == AffineScalarFunc
    assert y.nominal_value == 3.14
    assert y.std_dev == 0.01
    a = ufloat(-1, 0.001)
    y = 2 * x + 3 * x + 2 + a
    error_sources = y.error_components()
    assert len(error_sources) == 2
    assert error_sources[x] == 0.05
    assert error_sources[a] == 0.001
    assert y.derivatives[x] == 5
    x.std_dev = 1
    assert y.error_components()[x] == 5
    y = 2 * x
    try:
        y.std_dev = 1
    except AttributeError:
        pass
    else:
        raise Exception('std_dev should not be settable for calculated results')
    assert 10 / x.std_dev == x.std_score(10 + x.nominal_value)
    x.std_dev = 0
    try:
        x.std_score(1)
    except ValueError:
        pass