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
def test_powers_Rational():
    """Test Rational._eval_power"""
    assert S.Half ** S.Infinity == 0
    assert Rational(3, 2) ** S.Infinity is S.Infinity
    assert Rational(-1, 2) ** S.Infinity == 0
    assert Rational(-3, 2) ** S.Infinity == zoo
    assert Rational(3, 4) ** S.NaN is S.NaN
    assert Rational(-2, 3) ** S.NaN is S.NaN
    assert sqrt(Rational(4, 3)) == 2 * sqrt(3) / 3
    assert Rational(4, 3) ** Rational(3, 2) == 8 * sqrt(3) / 9
    assert sqrt(Rational(-4, 3)) == I * 2 * sqrt(3) / 3
    assert Rational(-4, 3) ** Rational(3, 2) == -I * 8 * sqrt(3) / 9
    assert Rational(27, 2) ** Rational(1, 3) == 3 * 2 ** Rational(2, 3) / 2
    assert Rational(5 ** 3, 8 ** 3) ** Rational(4, 3) == Rational(5 ** 4, 8 ** 4)
    assert sqrt(Rational(1, 4)) == S.Half
    assert sqrt(Rational(1, -4)) == I * S.Half
    assert sqrt(Rational(3, 4)) == sqrt(3) / 2
    assert sqrt(Rational(3, -4)) == I * sqrt(3) / 2
    assert Rational(5, 27) ** Rational(1, 3) == 5 ** Rational(1, 3) / 3
    assert sqrt(S.Half) == sqrt(2) / 2
    assert sqrt(Rational(-4, 7)) == I * sqrt(Rational(4, 7))
    assert Rational(-3, 2) ** Rational(-7, 3) == -4 * (-1) ** Rational(2, 3) * 2 ** Rational(1, 3) * 3 ** Rational(2, 3) / 27
    assert Rational(-3, 2) ** Rational(-2, 3) == -(-1) ** Rational(1, 3) * 2 ** Rational(2, 3) * 3 ** Rational(1, 3) / 3
    assert Rational(-3, 2) ** Rational(-10, 3) == 8 * (-1) ** Rational(2, 3) * 2 ** Rational(1, 3) * 3 ** Rational(2, 3) / 81
    assert abs(Pow(Rational(-2, 3), Rational(-7, 4)).n() - Pow(Rational(-2, 3), Rational(-7, 4), evaluate=False).n()) < 1e-16
    assert Rational(-2, 3) ** Rational(-2, 1) == Rational(9, 4)
    a = Rational(1, 10)
    assert a ** Float(a, 2) == Float(a, 2) ** Float(a, 2)
    assert Rational(-2, 3) ** Symbol('', even=True) == Rational(2, 3) ** Symbol('', even=True)