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
def test_Poly_degree():
    assert Poly(0, x).degree() is -oo
    assert Poly(1, x).degree() == 0
    assert Poly(x, x).degree() == 1
    assert Poly(0, x).degree(gen=0) is -oo
    assert Poly(1, x).degree(gen=0) == 0
    assert Poly(x, x).degree(gen=0) == 1
    assert Poly(0, x).degree(gen=x) is -oo
    assert Poly(1, x).degree(gen=x) == 0
    assert Poly(x, x).degree(gen=x) == 1
    assert Poly(0, x).degree(gen='x') is -oo
    assert Poly(1, x).degree(gen='x') == 0
    assert Poly(x, x).degree(gen='x') == 1
    raises(PolynomialError, lambda: Poly(1, x).degree(gen=1))
    raises(PolynomialError, lambda: Poly(1, x).degree(gen=y))
    raises(PolynomialError, lambda: Poly(1, x).degree(gen='y'))
    assert Poly(1, x, y).degree() == 0
    assert Poly(2 * y, x, y).degree() == 0
    assert Poly(x * y, x, y).degree() == 1
    assert Poly(1, x, y).degree(gen=x) == 0
    assert Poly(2 * y, x, y).degree(gen=x) == 0
    assert Poly(x * y, x, y).degree(gen=x) == 1
    assert Poly(1, x, y).degree(gen=y) == 0
    assert Poly(2 * y, x, y).degree(gen=y) == 1
    assert Poly(x * y, x, y).degree(gen=y) == 1
    assert degree(0, x) is -oo
    assert degree(1, x) == 0
    assert degree(x, x) == 1
    assert degree(x * y ** 2, x) == 1
    assert degree(x * y ** 2, y) == 2
    assert degree(x * y ** 2, z) == 0
    assert degree(pi) == 1
    raises(TypeError, lambda: degree(y ** 2 + x ** 3))
    raises(TypeError, lambda: degree(y ** 2 + x ** 3, 1))
    raises(PolynomialError, lambda: degree(x, 1.1))
    raises(PolynomialError, lambda: degree(x ** 2 / (x ** 3 + 1), x))
    assert degree(Poly(0, x), z) is -oo
    assert degree(Poly(1, x), z) == 0
    assert degree(Poly(x ** 2 + y ** 3, y)) == 3
    assert degree(Poly(y ** 2 + x ** 3, y, x), 1) == 3
    assert degree(Poly(y ** 2 + x ** 3, x), z) == 0
    assert degree(Poly(y ** 2 + x ** 3 + z ** 4, x), z) == 4