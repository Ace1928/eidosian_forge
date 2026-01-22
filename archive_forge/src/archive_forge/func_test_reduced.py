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
def test_reduced():
    f = 2 * x ** 4 + y ** 2 - x ** 2 + y ** 3
    G = [x ** 3 - x, y ** 3 - y]
    Q = [2 * x, 1]
    r = x ** 2 + y ** 2 + y
    assert reduced(f, G) == (Q, r)
    assert reduced(f, G, x, y) == (Q, r)
    H = groebner(G)
    assert H.reduce(f) == (Q, r)
    Q = [Poly(2 * x, x, y), Poly(1, x, y)]
    r = Poly(x ** 2 + y ** 2 + y, x, y)
    assert _strict_eq(reduced(f, G, polys=True), (Q, r))
    assert _strict_eq(reduced(f, G, x, y, polys=True), (Q, r))
    H = groebner(G, polys=True)
    assert _strict_eq(H.reduce(f), (Q, r))
    f = 2 * x ** 3 + y ** 3 + 3 * y
    G = groebner([x ** 2 + y ** 2 - 1, x * y - 2])
    Q = [x ** 2 - x * y ** 3 / 2 + x * y / 2 + y ** 6 / 4 - y ** 4 / 2 + y ** 2 / 4, -y ** 5 / 4 + y ** 3 / 2 + y * Rational(3, 4)]
    r = 0
    assert reduced(f, G) == (Q, r)
    assert G.reduce(f) == (Q, r)
    assert reduced(f, G, auto=False)[1] != 0
    assert G.reduce(f, auto=False)[1] != 0
    assert G.contains(f) is True
    assert G.contains(f + 1) is False
    assert reduced(1, [1], x) == ([1], 0)
    raises(ComputationFailed, lambda: reduced(1, [1]))