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
def test_Poly_eval():
    assert Poly(0, x).eval(7) == 0
    assert Poly(1, x).eval(7) == 1
    assert Poly(x, x).eval(7) == 7
    assert Poly(0, x).eval(0, 7) == 0
    assert Poly(1, x).eval(0, 7) == 1
    assert Poly(x, x).eval(0, 7) == 7
    assert Poly(0, x).eval(x, 7) == 0
    assert Poly(1, x).eval(x, 7) == 1
    assert Poly(x, x).eval(x, 7) == 7
    assert Poly(0, x).eval('x', 7) == 0
    assert Poly(1, x).eval('x', 7) == 1
    assert Poly(x, x).eval('x', 7) == 7
    raises(PolynomialError, lambda: Poly(1, x).eval(1, 7))
    raises(PolynomialError, lambda: Poly(1, x).eval(y, 7))
    raises(PolynomialError, lambda: Poly(1, x).eval('y', 7))
    assert Poly(123, x, y).eval(7) == Poly(123, y)
    assert Poly(2 * y, x, y).eval(7) == Poly(2 * y, y)
    assert Poly(x * y, x, y).eval(7) == Poly(7 * y, y)
    assert Poly(123, x, y).eval(x, 7) == Poly(123, y)
    assert Poly(2 * y, x, y).eval(x, 7) == Poly(2 * y, y)
    assert Poly(x * y, x, y).eval(x, 7) == Poly(7 * y, y)
    assert Poly(123, x, y).eval(y, 7) == Poly(123, x)
    assert Poly(2 * y, x, y).eval(y, 7) == Poly(14, x)
    assert Poly(x * y, x, y).eval(y, 7) == Poly(7 * x, x)
    assert Poly(x * y + y, x, y).eval({x: 7}) == Poly(8 * y, y)
    assert Poly(x * y + y, x, y).eval({y: 7}) == Poly(7 * x + 7, x)
    assert Poly(x * y + y, x, y).eval({x: 6, y: 7}) == 49
    assert Poly(x * y + y, x, y).eval({x: 7, y: 6}) == 48
    assert Poly(x * y + y, x, y).eval((6, 7)) == 49
    assert Poly(x * y + y, x, y).eval([6, 7]) == 49
    assert Poly(x + 1, domain='ZZ').eval(S.Half) == Rational(3, 2)
    assert Poly(x + 1, domain='ZZ').eval(sqrt(2)) == sqrt(2) + 1
    raises(ValueError, lambda: Poly(x * y + y, x, y).eval((6, 7, 8)))
    raises(DomainError, lambda: Poly(x + 1, domain='ZZ').eval(S.Half, auto=False))
    alpha = Symbol('alpha')
    result = (2 * alpha * z - 2 * alpha + z ** 2 + 3) / (z ** 2 - 2 * z + 1)
    f = Poly(x ** 2 + (alpha - 1) * x - alpha + 1, x, domain='ZZ[alpha]')
    assert f.eval((z + 1) / (z - 1)) == result
    g = Poly(x ** 2 + (alpha - 1) * x - alpha + 1, x, y, domain='ZZ[alpha]')
    assert g.eval((z + 1) / (z - 1)) == Poly(result, y, domain='ZZ(alpha,z)')