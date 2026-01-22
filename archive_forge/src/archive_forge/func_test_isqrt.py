import numbers as nums
import decimal
from sympy.concrete.summations import Sum
from sympy.core import (EulerGamma, Catalan, TribonacciConstant,
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import (mpf_norm, mod_inverse, igcd, seterr,
from sympy.core.power import Pow
from sympy.core.relational import Ge, Gt, Le, Lt
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.integers import floor
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.polys.domains.realfield import RealField
from sympy.printing.latex import latex
from sympy.printing.repr import srepr
from sympy.simplify import simplify
from sympy.core.power import integer_nthroot, isqrt, integer_log
from sympy.polys.domains.groundtypes import PythonRational
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow
from mpmath import mpf
from mpmath.rational import mpq
import mpmath
from sympy.core import numbers
def test_isqrt():
    from math import sqrt as _sqrt
    limit = 4503599761588223
    assert int(_sqrt(limit)) == integer_nthroot(limit, 2)[0]
    assert int(_sqrt(limit + 1)) != integer_nthroot(limit + 1, 2)[0]
    assert isqrt(limit + 1) == integer_nthroot(limit + 1, 2)[0]
    assert isqrt(limit + S.Half) == integer_nthroot(limit, 2)[0]
    assert isqrt(limit + 1 + S.Half) == integer_nthroot(limit + 1, 2)[0]
    assert isqrt(limit + 2 + S.Half) == integer_nthroot(limit + 2, 2)[0]
    assert isqrt(4503599761588224) == 67108864
    assert isqrt(9999999999999999) == 99999999
    raises(ValueError, lambda: isqrt(-1))
    raises(ValueError, lambda: isqrt(-10 ** 1000))
    raises(ValueError, lambda: isqrt(Rational(-1, 2)))
    tiny = Rational(1, 10 ** 1000)
    raises(ValueError, lambda: isqrt(-tiny))
    assert isqrt(1 - tiny) == 0
    assert isqrt(4503599761588224 - tiny) == 67108864
    assert isqrt(10 ** 100 - tiny) == 10 ** 50 - 1
    from sympy.core import power
    old_sqrt = power._sqrt
    power._sqrt = lambda x: 2.999999999
    try:
        assert isqrt(9) == 3
        assert isqrt(10000) == 100
    finally:
        power._sqrt = old_sqrt