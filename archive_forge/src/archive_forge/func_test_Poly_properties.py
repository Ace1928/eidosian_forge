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
def test_Poly_properties():
    assert Poly(0, x).is_zero is True
    assert Poly(1, x).is_zero is False
    assert Poly(1, x).is_one is True
    assert Poly(2, x).is_one is False
    assert Poly(x - 1, x).is_sqf is True
    assert Poly((x - 1) ** 2, x).is_sqf is False
    assert Poly(x - 1, x).is_monic is True
    assert Poly(2 * x - 1, x).is_monic is False
    assert Poly(3 * x + 2, x).is_primitive is True
    assert Poly(4 * x + 2, x).is_primitive is False
    assert Poly(1, x).is_ground is True
    assert Poly(x, x).is_ground is False
    assert Poly(x + y + z + 1).is_linear is True
    assert Poly(x * y * z + 1).is_linear is False
    assert Poly(x * y + z + 1).is_quadratic is True
    assert Poly(x * y * z + 1).is_quadratic is False
    assert Poly(x * y).is_monomial is True
    assert Poly(x * y + 1).is_monomial is False
    assert Poly(x ** 2 + x * y).is_homogeneous is True
    assert Poly(x ** 3 + x * y).is_homogeneous is False
    assert Poly(x).is_univariate is True
    assert Poly(x * y).is_univariate is False
    assert Poly(x * y).is_multivariate is True
    assert Poly(x).is_multivariate is False
    assert Poly(x ** 16 + x ** 14 - x ** 10 + x ** 8 - x ** 6 + x ** 2 + 1).is_cyclotomic is False
    assert Poly(x ** 16 + x ** 14 - x ** 10 - x ** 8 - x ** 6 + x ** 2 + 1).is_cyclotomic is True