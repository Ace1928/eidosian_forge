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
def test_same_root():
    f = Poly(x ** 4 + x ** 3 + x ** 2 + x + 1)
    eq = f.same_root
    r0 = exp(2 * I * pi / 5)
    assert [i for i, r in enumerate(f.all_roots()) if eq(r, r0)] == [3]
    raises(PolynomialError, lambda: Poly(x + 1, domain=QQ).same_root(0, 0))
    raises(DomainError, lambda: Poly(x ** 2 + 1, domain=FF(7)).same_root(0, 0))
    raises(DomainError, lambda: Poly(x ** 2 + 1, domain=ZZ_I).same_root(0, 0))
    raises(DomainError, lambda: Poly(y * x ** 2 + 1, domain=ZZ[y]).same_root(0, 0))
    raises(MultivariatePolynomialError, lambda: Poly(x * y + 1, domain=ZZ).same_root(0, 0))