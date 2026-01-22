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
def test_wrap_with_kwargs():
    """
    Tests wrap() on functions with keyword arguments.

    Includes both wrapping a function that takes optional keyword
    arguments and calling a wrapped function with keyword arguments
    (optional or not).
    """

    def f_auto_unc(x, y, *args, **kwargs):
        return x + umath.sin(y) + 2 * args[0] + 3 * kwargs['t']

    def f(x, y, *args, **kwargs):
        for value in [x, y] + list(args) + list(kwargs.values()):
            assert not isinstance(value, uncert_core.UFloat)
        return f_auto_unc(x, y, *args, **kwargs)
    f_wrapped = uncert_core.wrap(f)
    x = ufloat(1, 0.1)
    y = ufloat(10, 0.11)
    z = ufloat(100, 0.111)
    t = ufloat(0.1, 0.1111)
    assert ufloats_close(f_wrapped(x, y, z, t=t), f_auto_unc(x, y, z, t=t), tolerance=1e-05)
    f_wrapped2 = uncert_core.wrap(f, [None, lambda x, y, *args, **kwargs: math.cos(y)])
    assert f_wrapped2(x, y, z, t=t).derivatives[y] == f_auto_unc(x, y, z, t=t).derivatives[y]
    f_wrapped3 = uncert_core.wrap(f, [None, None, lambda x, y, *args, **kwargs: 2], {'t': lambda x, y, *args, **kwargs: 3})
    assert f_wrapped3(x, y, z, t=t).derivatives[z] == f_auto_unc(x, y, z, t=t).derivatives[z]
    assert f_wrapped3(x, y, z, t=t).derivatives[t] == f_auto_unc(x, y, z, t=t).derivatives[t]

    class FunctionCalled(Exception):
        """
        Raised to signal that a function is indeed called.
        """
        pass

    def failing_func(x, y, *args, **kwargs):
        raise FunctionCalled
    f_wrapped4 = uncert_core.wrap(f, [None, failing_func], {'t': failing_func})
    try:
        f_wrapped4(x, 3.14, z, t=t)
    except FunctionCalled:
        pass
    else:
        raise Exception('User-supplied derivative should be called')
    try:
        f_wrapped4(x, y, z, t=3.14)
    except FunctionCalled:
        pass
    else:
        raise Exception('User-supplied derivative should be called')
    try:
        f_wrapped4(x, 3.14, z, t=3.14)
    except FunctionCalled:
        raise Exception('User-supplied derivative should *not* be called')