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
def test_Poly_coeff():
    assert Poly(0, x).coeff_monomial(1) == 0
    assert Poly(0, x).coeff_monomial(x) == 0
    assert Poly(1, x).coeff_monomial(1) == 1
    assert Poly(1, x).coeff_monomial(x) == 0
    assert Poly(x ** 8, x).coeff_monomial(1) == 0
    assert Poly(x ** 8, x).coeff_monomial(x ** 7) == 0
    assert Poly(x ** 8, x).coeff_monomial(x ** 8) == 1
    assert Poly(x ** 8, x).coeff_monomial(x ** 9) == 0
    assert Poly(3 * x * y ** 2 + 1, x, y).coeff_monomial(1) == 1
    assert Poly(3 * x * y ** 2 + 1, x, y).coeff_monomial(x * y ** 2) == 3
    p = Poly(24 * x * y * exp(8) + 23 * x, x, y)
    assert p.coeff_monomial(x) == 23
    assert p.coeff_monomial(y) == 0
    assert p.coeff_monomial(x * y) == 24 * exp(8)
    assert p.as_expr().coeff(x) == 24 * y * exp(8) + 23
    raises(NotImplementedError, lambda: p.coeff(x))
    raises(ValueError, lambda: Poly(x + 1).coeff_monomial(0))
    raises(ValueError, lambda: Poly(x + 1).coeff_monomial(3 * x))
    raises(ValueError, lambda: Poly(x + 1).coeff_monomial(3 * x * y))