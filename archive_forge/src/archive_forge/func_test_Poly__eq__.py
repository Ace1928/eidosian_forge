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
def test_Poly__eq__():
    assert (Poly(x, x) == Poly(x, x)) is True
    assert (Poly(x, x, domain=QQ) == Poly(x, x)) is False
    assert (Poly(x, x) == Poly(x, x, domain=QQ)) is False
    assert (Poly(x, x, domain=ZZ[a]) == Poly(x, x)) is False
    assert (Poly(x, x) == Poly(x, x, domain=ZZ[a])) is False
    assert (Poly(x * y, x, y) == Poly(x, x)) is False
    assert (Poly(x, x, y) == Poly(x, x)) is False
    assert (Poly(x, x) == Poly(x, x, y)) is False
    assert (Poly(x ** 2 + 1, x) == Poly(y ** 2 + 1, y)) is False
    assert (Poly(y ** 2 + 1, y) == Poly(x ** 2 + 1, x)) is False
    f = Poly(x, x, domain=ZZ)
    g = Poly(x, x, domain=QQ)
    assert f.eq(g) is False
    assert f.ne(g) is True
    assert f.eq(g, strict=True) is False
    assert f.ne(g, strict=True) is True
    t0 = Symbol('t0')
    f = Poly((t0 / 2 + x ** 2) * t ** 2 - x ** 2 * t, t, domain='QQ[x,t0]')
    g = Poly((t0 / 2 + x ** 2) * t ** 2 - x ** 2 * t, t, domain='ZZ(x,t0)')
    assert (f == g) is False