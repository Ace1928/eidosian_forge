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
def test_sqf():
    f = x ** 5 - x ** 3 - x ** 2 + 1
    g = x ** 3 + 2 * x ** 2 + 2 * x + 1
    h = x - 1
    p = x ** 4 + x ** 3 - x - 1
    F, G, H, P = map(Poly, (f, g, h, p))
    assert F.sqf_part() == P
    assert sqf_part(f) == p
    assert sqf_part(f, x) == p
    assert sqf_part(f, (x,)) == p
    assert sqf_part(F) == P
    assert sqf_part(f, polys=True) == P
    assert sqf_part(F, polys=False) == p
    assert F.sqf_list() == (1, [(G, 1), (H, 2)])
    assert sqf_list(f) == (1, [(g, 1), (h, 2)])
    assert sqf_list(f, x) == (1, [(g, 1), (h, 2)])
    assert sqf_list(f, (x,)) == (1, [(g, 1), (h, 2)])
    assert sqf_list(F) == (1, [(G, 1), (H, 2)])
    assert sqf_list(f, polys=True) == (1, [(G, 1), (H, 2)])
    assert sqf_list(F, polys=False) == (1, [(g, 1), (h, 2)])
    assert F.sqf_list_include() == [(G, 1), (H, 2)]
    raises(ComputationFailed, lambda: sqf_part(4))
    assert sqf(1) == 1
    assert sqf_list(1) == (1, [])
    assert sqf((2 * x ** 2 + 2) ** 7) == 128 * (x ** 2 + 1) ** 7
    assert sqf(f) == g * h ** 2
    assert sqf(f, x) == g * h ** 2
    assert sqf(f, (x,)) == g * h ** 2
    d = x ** 2 + y ** 2
    assert sqf(f / d) == g * h ** 2 / d
    assert sqf(f / d, x) == g * h ** 2 / d
    assert sqf(f / d, (x,)) == g * h ** 2 / d
    assert sqf(x - 1) == x - 1
    assert sqf(-x - 1) == -x - 1
    assert sqf(x - 1) == x - 1
    assert sqf(6 * x - 10) == Mul(2, 3 * x - 5, evaluate=False)
    assert sqf((6 * x - 10) / (3 * x - 6)) == Rational(2, 3) * ((3 * x - 5) / (x - 2))
    assert sqf(Poly(x ** 2 - 2 * x + 1)) == (x - 1) ** 2
    f = 3 + x - x * (1 + x) + x ** 2
    assert sqf(f) == 3
    f = (x ** 2 + 2 * x + 1) ** 20000000000
    assert sqf(f) == (x + 1) ** 40000000000
    assert sqf_list(f) == (1, [(x + 1, 40000000000)])