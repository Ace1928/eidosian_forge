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
def test_wrapped_func_args_kwargs():
    """
    Wrap a function that takes positional-or-keyword, var-positional
    and var-keyword parameters.
    """

    def f_auto_unc(x, y, *args, **kwargs):
        return 2 * x + umath.sin(y) + 4 * args[1] + 3 * kwargs['z']

    def f(x, y, *args, **kwargs):
        assert not any((isinstance(value, uncert_core.UFloat) for value in [x, y] + list(args) + list(kwargs.values())))
        return f_auto_unc(x, y, *args, **kwargs)
    x = uncert_core.ufloat(1, 0.1)
    y = uncert_core.ufloat(10, 2)
    t = uncert_core.ufloat(1000, 4)
    s = 'string arg'
    z = uncert_core.ufloat(100, 3)
    args = [s, t, s]
    kwargs = {'u': s, 'z': z}
    f_wrapped = uncert_core.wrap(f)
    assert ufloats_close(f_auto_unc(x, y, *args, **kwargs), f_wrapped(x, y, *args, **kwargs), tolerance=1e-05)
    f_wrapped = uncert_core.wrap(f, [None, None, None, lambda x, y, *args, **kwargs: 4])
    assert ufloats_close(f_auto_unc(x, y, *args, **kwargs), f_wrapped(x, y, *args, **kwargs), tolerance=1e-05)
    f_wrapped = uncert_core.wrap(f, [None], {'z': None})
    assert ufloats_close(f_auto_unc(x, y, *args, **kwargs), f_wrapped(x, y, *args, **kwargs), tolerance=1e-05)
    f_wrapped = uncert_core.wrap(f, [None], {'z': lambda x, y, *args, **kwargs: 3})
    assert ufloats_close(f_auto_unc(x, y, *args, **kwargs), f_wrapped(x, y, *args, **kwargs), tolerance=1e-05)
    f_wrapped = uncert_core.wrap(f, [lambda x, y, *args, **kwargs: 2, lambda x, y, *args, **kwargs: math.cos(y)], {'z:': lambda x, y, *args, **kwargs: 3})
    assert ufloats_close(f_auto_unc(x, y, *args, **kwargs), f_wrapped(x, y, *args, **kwargs), tolerance=1e-05)
    f_wrapped = uncert_core.wrap(f, [lambda x, y, *args, **kwargs: 2])
    assert ufloats_close(f_auto_unc(x, y, *args, **kwargs), f_wrapped(x, y, *args, **kwargs), tolerance=1e-05)