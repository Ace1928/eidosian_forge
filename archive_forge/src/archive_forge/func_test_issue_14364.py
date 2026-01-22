import pickle
from sympy.polys.polytools import (
from sympy.polys.polyerrors import (
from sympy.polys.polyclasses import DMP
from sympy.polys.fields import field
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX
from sympy.polys.domains.realfield import RealField
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.orderings import lex, grlex, grevlex
from sympy.combinatorics.galois import S4TransitiveSubgroups
from sympy.core.add import Add
from sympy.core.basic import _aresame
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, diff, expand)
from sympy.core.mul import _keep_coeff, Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys.rootoftools import rootof
from sympy.simplify.simplify import signsimp
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.testing.pytest import raises, warns_deprecated_sympy, warns
from sympy.abc import a, b, c, d, p, q, t, w, x, y, z
def test_issue_14364():
    assert gcd(S(6) * (1 + sqrt(3)) / 5, S(3) * (1 + sqrt(3)) / 10) == Rational(3, 10) * (1 + sqrt(3))
    assert gcd(sqrt(5) * Rational(4, 7), sqrt(5) * Rational(2, 3)) == sqrt(5) * Rational(2, 21)
    assert lcm(Rational(2, 3) * sqrt(3), Rational(5, 6) * sqrt(3)) == S(10) * sqrt(3) / 3
    assert lcm(3 * sqrt(3), 4 / sqrt(3)) == 12 * sqrt(3)
    assert lcm(S(5) * (1 + 2 ** Rational(1, 3)) / 6, S(3) * (1 + 2 ** Rational(1, 3)) / 8) == Rational(15, 2) * (1 + 2 ** Rational(1, 3))
    assert gcd(Rational(2, 3) * sqrt(3), Rational(5, 6) / sqrt(3)) == sqrt(3) / 18
    assert gcd(S(4) * sqrt(13) / 7, S(3) * sqrt(13) / 14) == sqrt(13) / 14
    assert gcd([S(2) * sqrt(47) / 7, S(6) * sqrt(47) / 5, S(8) * sqrt(47) / 5]) == sqrt(47) * Rational(2, 35)
    assert gcd([S(6) * (1 + sqrt(7)) / 5, S(2) * (1 + sqrt(7)) / 7, S(4) * (1 + sqrt(7)) / 13]) == (1 + sqrt(7)) * Rational(2, 455)
    assert lcm((Rational(7, 2) / sqrt(15), Rational(5, 6) / sqrt(15), Rational(5, 8) / sqrt(15))) == Rational(35, 2) / sqrt(15)
    assert lcm([S(5) * (2 + 2 ** Rational(5, 7)) / 6, S(7) * (2 + 2 ** Rational(5, 7)) / 2, S(13) * (2 + 2 ** Rational(5, 7)) / 4]) == Rational(455, 2) * (2 + 2 ** Rational(5, 7))