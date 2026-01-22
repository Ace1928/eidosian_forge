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
def test_powers_Integer():
    """Test Integer._eval_power"""
    assert S.One ** S.Infinity is S.NaN
    assert S.NegativeOne ** S.Infinity is S.NaN
    assert S(2) ** S.Infinity is S.Infinity
    assert S(-2) ** S.Infinity == zoo
    assert S(0) ** S.Infinity is S.Zero
    assert S.One ** S.NaN is S.NaN
    assert S.NegativeOne ** S.NaN is S.NaN
    assert S.NegativeOne ** Rational(6, 5) == -(-1) ** (S.One / 5)
    assert sqrt(S(4)) == 2
    assert sqrt(S(-4)) == I * 2
    assert S(16) ** Rational(1, 4) == 2
    assert S(-16) ** Rational(1, 4) == 2 * (-1) ** Rational(1, 4)
    assert S(9) ** Rational(3, 2) == 27
    assert S(-9) ** Rational(3, 2) == -27 * I
    assert S(27) ** Rational(2, 3) == 9
    assert S(-27) ** Rational(2, 3) == 9 * S.NegativeOne ** Rational(2, 3)
    assert (-2) ** Rational(-2, 1) == Rational(1, 4)
    assert sqrt(-3) == I * sqrt(3)
    assert 3 ** Rational(3, 2) == 3 * sqrt(3)
    assert (-3) ** Rational(3, 2) == -3 * sqrt(-3)
    assert (-3) ** Rational(5, 2) == 9 * I * sqrt(3)
    assert (-3) ** Rational(7, 2) == -I * 27 * sqrt(3)
    assert 2 ** Rational(3, 2) == 2 * sqrt(2)
    assert 2 ** Rational(-3, 2) == sqrt(2) / 4
    assert 81 ** Rational(2, 3) == 9 * S(3) ** Rational(2, 3)
    assert (-81) ** Rational(2, 3) == 9 * S(-3) ** Rational(2, 3)
    assert (-3) ** Rational(-7, 3) == -(-1) ** Rational(2, 3) * 3 ** Rational(2, 3) / 27
    assert (-3) ** Rational(-2, 3) == -(-1) ** Rational(1, 3) * 3 ** Rational(1, 3) / 3
    assert sqrt(6) + sqrt(24) == 3 * sqrt(6)
    assert sqrt(2) * sqrt(3) == sqrt(6)
    x = Symbol('x')
    assert sqrt(49 * x) == 7 * sqrt(x)
    assert sqrt((3 - sqrt(pi)) ** 2) == 3 - sqrt(pi)
    assert (2 ** 64 + 1) ** Rational(4, 3)
    assert (2 ** 64 + 1) ** Rational(17, 25)
    assert (-3) ** Rational(-7, 3) == -(-1) ** Rational(2, 3) * 3 ** Rational(2, 3) / 27
    assert (-3) ** Rational(-2, 3) == -(-1) ** Rational(1, 3) * 3 ** Rational(1, 3) / 3
    assert (-2) ** Rational(-10, 3) == (-1) ** Rational(2, 3) * 2 ** Rational(2, 3) / 16
    assert abs(Pow(-2, Rational(-10, 3)).n() - Pow(-2, Rational(-10, 3), evaluate=False).n()) < 1e-16
    assert (-8) ** Rational(2, 5) == 2 * (-1) ** Rational(2, 5) * 2 ** Rational(1, 5)
    assert (-4) ** Rational(9, 5) == -8 * (-1) ** Rational(4, 5) * 2 ** Rational(3, 5)
    assert S(1234).factors() == {617: 1, 2: 1}
    assert Rational(2 * 3, 3 * 5 * 7).factors() == {2: 1, 5: -1, 7: -1}
    from sympy.ntheory.generate import nextprime
    n = nextprime(2 ** 15)
    assert sqrt(n ** 2) == n
    assert sqrt(n ** 3) == n * sqrt(n)
    assert sqrt(4 * n) == 2 * sqrt(n)
    assert (2 ** 4 * 3) ** Rational(1, 6) == 2 ** Rational(2, 3) * 3 ** Rational(1, 6)
    assert (2 ** 4 * 3) ** Rational(5, 6) == 8 * 2 ** Rational(1, 3) * 3 ** Rational(5, 6)
    assert 2 ** Rational(1, 3) * 3 ** Rational(1, 4) * 6 ** Rational(1, 5) == 2 ** Rational(8, 15) * 3 ** Rational(9, 20)
    assert sqrt(8) * 24 ** Rational(1, 3) * 6 ** Rational(1, 5) == 4 * 2 ** Rational(7, 10) * 3 ** Rational(8, 15)
    assert sqrt(8) * (-24) ** Rational(1, 3) * (-6) ** Rational(1, 5) == 4 * (-3) ** Rational(8, 15) * 2 ** Rational(7, 10)
    assert 2 ** Rational(1, 3) * 2 ** Rational(8, 9) == 2 * 2 ** Rational(2, 9)
    assert 2 ** Rational(2, 3) * 6 ** Rational(1, 3) == 2 * 3 ** Rational(1, 3)
    assert 2 ** Rational(2, 3) * 6 ** Rational(8, 9) == 2 * 2 ** Rational(5, 9) * 3 ** Rational(8, 9)
    assert (-2) ** Rational(2, S(3)) * (-4) ** Rational(1, S(3)) == -2 * 2 ** Rational(1, 3)
    assert 3 * Pow(3, 2, evaluate=False) == 3 ** 3
    assert 3 * Pow(3, Rational(-1, 3), evaluate=False) == 3 ** Rational(2, 3)
    assert (-2) ** Rational(1, 3) * (-3) ** Rational(1, 4) * (-5) ** Rational(5, 6) == -(-1) ** Rational(5, 12) * 2 ** Rational(1, 3) * 3 ** Rational(1, 4) * 5 ** Rational(5, 6)
    assert Integer(-2) ** Symbol('', even=True) == Integer(2) ** Symbol('', even=True)
    assert (-1) ** Float(0.5) == 1.0 * I