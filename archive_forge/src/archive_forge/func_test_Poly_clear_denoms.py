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
def test_Poly_clear_denoms():
    coeff, poly = Poly(x + 2, x).clear_denoms()
    assert coeff == 1 and poly == Poly(x + 2, x, domain='ZZ') and (poly.get_domain() == ZZ)
    coeff, poly = Poly(x / 2 + 1, x).clear_denoms()
    assert coeff == 2 and poly == Poly(x + 2, x, domain='QQ') and (poly.get_domain() == QQ)
    coeff, poly = Poly(x / 2 + 1, x).clear_denoms(convert=True)
    assert coeff == 2 and poly == Poly(x + 2, x, domain='ZZ') and (poly.get_domain() == ZZ)
    coeff, poly = Poly(x / y + 1, x).clear_denoms(convert=True)
    assert coeff == y and poly == Poly(x + y, x, domain='ZZ[y]') and (poly.get_domain() == ZZ[y])
    coeff, poly = Poly(x / 3 + sqrt(2), x, domain='EX').clear_denoms()
    assert coeff == 3 and poly == Poly(x + 3 * sqrt(2), x, domain='EX') and (poly.get_domain() == EX)
    coeff, poly = Poly(x / 3 + sqrt(2), x, domain='EX').clear_denoms(convert=True)
    assert coeff == 3 and poly == Poly(x + 3 * sqrt(2), x, domain='EX') and (poly.get_domain() == EX)