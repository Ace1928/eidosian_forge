import string
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import (diff, expand_func)
from sympy.core import (EulerGamma, TribonacciConstant)
from sympy.core.numbers import (Float, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.numbers import carmichael
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.integers import floor
from sympy.polys.polytools import cancel
from sympy.series.limits import limit, Limit
from sympy.series.order import O
from sympy.functions import (
from sympy.functions.combinatorial.numbers import _nT
from sympy.core.expr import unchanged
from sympy.core.numbers import GoldenRatio, Integer
from sympy.testing.pytest import raises, nocache_fail, warns_deprecated_sympy
from sympy.abc import x
def test_carmichael():
    assert carmichael.find_carmichael_numbers_in_range(0, 561) == []
    assert carmichael.find_carmichael_numbers_in_range(561, 562) == [561]
    assert carmichael.find_carmichael_numbers_in_range(561, 1105) == carmichael.find_carmichael_numbers_in_range(561, 562)
    assert carmichael.find_first_n_carmichaels(5) == [561, 1105, 1729, 2465, 2821]
    raises(ValueError, lambda: carmichael.is_carmichael(-2))
    raises(ValueError, lambda: carmichael.find_carmichael_numbers_in_range(-2, 2))
    raises(ValueError, lambda: carmichael.find_carmichael_numbers_in_range(22, 2))
    with warns_deprecated_sympy():
        assert carmichael.is_prime(2821) == False