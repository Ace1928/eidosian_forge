from __future__ import division
from __future__ import absolute_import
import sys
import math
from uncertainties import ufloat
import uncertainties.core as uncert_core
import uncertainties.umath_core as umath_core
from . import test_uncertainties
def test_math_module():
    """Operations with the math module"""
    x = ufloat(-1.5, 0.1)
    assert (x ** 2).nominal_value == 2.25
    assert isinstance(umath_core.sin(3), float)
    assert umath_core.factorial(4) == 24
    assert umath_core.fsum([x, x]).nominal_value == -3
    for name in umath_core.locally_cst_funcs:
        try:
            func = getattr(umath_core, name)
        except AttributeError:
            continue
        assert func(x) == func(x.nominal_value)
        assert type(func(x)) == type(func(x.nominal_value))
    try:
        math.log(0)
    except Exception as err_math:
        err_math_args = err_math.args
        exception_class = err_math.__class__
    try:
        umath_core.log(0)
    except exception_class as err_ufloat:
        assert err_math_args == err_ufloat.args
    else:
        raise Exception('%s exception expected' % exception_class.__name__)
    try:
        umath_core.log(ufloat(0, 0))
    except exception_class as err_ufloat:
        assert err_math_args == err_ufloat.args
    else:
        raise Exception('%s exception expected' % exception_class.__name__)
    try:
        umath_core.log(ufloat(0, 1))
    except exception_class as err_ufloat:
        assert err_math_args == err_ufloat.args
    else:
        raise Exception('%s exception expected' % exception_class.__name__)