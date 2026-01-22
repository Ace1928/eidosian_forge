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
def test_wrapped_func():
    """
    Test uncertainty-aware functions obtained through wrapping.
    """

    def f_auto_unc(angle, *list_var):
        return umath.cos(angle) + sum(list_var)

    def f(angle, *list_var):
        assert not isinstance(angle, uncert_core.UFloat)
        assert not any((isinstance(arg, uncert_core.UFloat) for arg in list_var))
        return f_auto_unc(angle, *list_var)
    f_wrapped = uncert_core.wrap(f)
    my_list = [1, 2, 3]
    assert f_wrapped(0, *my_list) == f(0, *my_list)
    assert type(f_wrapped(0, *my_list)) == type(f(0, *my_list))
    angle = uncert_core.ufloat(1, 0.1)
    list_value = uncert_core.ufloat(3, 0.2)
    assert ufloats_close(f_wrapped(angle, *[1, angle]), f_auto_unc(angle, *[1, angle]))
    assert ufloats_close(f_wrapped(angle, *[list_value, angle]), f_auto_unc(angle, *[list_value, angle]))

    def f(x, y, z, t, u):
        return x + 2 * z + 3 * t + 4 * u
    f_wrapped = uncert_core.wrap(f, [lambda *args: 1, None, lambda *args: 2, None])
    assert f_wrapped(10, 'string argument', 1, 0, 0) == 12
    x = uncert_core.ufloat(10, 1)
    assert numbers_close(f_wrapped(x, 'string argument', x, x, x).std_dev, (1 + 2 + 3 + 4) * x.std_dev)