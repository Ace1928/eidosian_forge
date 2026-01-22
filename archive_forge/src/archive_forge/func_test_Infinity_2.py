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
def test_Infinity_2():
    x = Symbol('x')
    assert oo * x != oo
    assert oo * (pi - 1) is oo
    assert oo * (1 - pi) is -oo
    assert -oo * x != -oo
    assert -oo * (pi - 1) is -oo
    assert -oo * (1 - pi) is oo
    assert (-1) ** S.NaN is S.NaN
    assert oo - _inf is S.NaN
    assert oo + _ninf is S.NaN
    assert oo * 0 is S.NaN
    assert oo / _inf is S.NaN
    assert oo / _ninf is S.NaN
    assert oo ** S.NaN is S.NaN
    assert -oo + _inf is S.NaN
    assert -oo - _ninf is S.NaN
    assert -oo * S.NaN is S.NaN
    assert -oo * 0 is S.NaN
    assert -oo / _inf is S.NaN
    assert -oo / _ninf is S.NaN
    assert -oo / S.NaN is S.NaN
    assert abs(-oo) is oo
    assert all(((-oo) ** i is S.NaN for i in (oo, -oo, S.NaN)))
    assert (-oo) ** 3 is -oo
    assert (-oo) ** 2 is oo
    assert abs(S.ComplexInfinity) is oo