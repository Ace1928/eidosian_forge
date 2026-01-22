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
def test_groebner():
    assert groebner([], x, y, z) == []
    assert groebner([x ** 2 + 1, y ** 4 * x + x ** 3], x, y, order='lex') == [1 + x ** 2, -1 + y ** 4]
    assert groebner([x ** 2 + 1, y ** 4 * x + x ** 3, x * y * z ** 3], x, y, z, order='grevlex') == [-1 + y ** 4, z ** 3, 1 + x ** 2]
    assert groebner([x ** 2 + 1, y ** 4 * x + x ** 3], x, y, order='lex', polys=True) == [Poly(1 + x ** 2, x, y), Poly(-1 + y ** 4, x, y)]
    assert groebner([x ** 2 + 1, y ** 4 * x + x ** 3, x * y * z ** 3], x, y, z, order='grevlex', polys=True) == [Poly(-1 + y ** 4, x, y, z), Poly(z ** 3, x, y, z), Poly(1 + x ** 2, x, y, z)]
    assert groebner([x ** 3 - 1, x ** 2 - 1]) == [x - 1]
    assert groebner([Eq(x ** 3, 1), Eq(x ** 2, 1)]) == [x - 1]
    F = [3 * x ** 2 + y * z - 5 * x - 1, 2 * x + 3 * x * y + y ** 2, x - 3 * y + x * z - 2 * z ** 2]
    f = z ** 9 - x ** 2 * y ** 3 - 3 * x * y ** 2 * z + 11 * y * z ** 2 + x ** 2 * z ** 2 - 5
    G = groebner(F, x, y, z, modulus=7, symmetric=False)
    assert G == [1 + x + y + 3 * z + 2 * z ** 2 + 2 * z ** 3 + 6 * z ** 4 + z ** 5, 1 + 3 * y + y ** 2 + 6 * z ** 2 + 3 * z ** 3 + 3 * z ** 4 + 3 * z ** 5 + 4 * z ** 6, 1 + 4 * y + 4 * z + y * z + 4 * z ** 3 + z ** 4 + z ** 6, 6 + 6 * z + z ** 2 + 4 * z ** 3 + 3 * z ** 4 + 6 * z ** 5 + 3 * z ** 6 + z ** 7]
    Q, r = reduced(f, G, x, y, z, modulus=7, symmetric=False, polys=True)
    assert sum([q * g for q, g in zip(Q, G.polys)], r) == Poly(f, modulus=7)
    F = [x * y - 2 * y, 2 * y ** 2 - x ** 2]
    assert groebner(F, x, y, order='grevlex') == [y ** 3 - 2 * y, x ** 2 - 2 * y ** 2, x * y - 2 * y]
    assert groebner(F, y, x, order='grevlex') == [x ** 3 - 2 * x ** 2, -x ** 2 + 2 * y ** 2, x * y - 2 * y]
    assert groebner(F, order='grevlex', field=True) == [y ** 3 - 2 * y, x ** 2 - 2 * y ** 2, x * y - 2 * y]
    assert groebner([1], x) == [1]
    assert groebner([x ** 2 + 2.0 * y], x, y) == [1.0 * x ** 2 + 2.0 * y]
    raises(ComputationFailed, lambda: groebner([1]))
    assert groebner([x ** 2 - 1, x ** 3 + 1], method='buchberger') == [x + 1]
    assert groebner([x ** 2 - 1, x ** 3 + 1], method='f5b') == [x + 1]
    raises(ValueError, lambda: groebner([x, y], method='unknown'))