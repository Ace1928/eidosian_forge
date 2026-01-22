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
def test_keep_coeff():
    u = Mul(2, x + 1, evaluate=False)
    assert _keep_coeff(S.One, x) == x
    assert _keep_coeff(S.NegativeOne, x) == -x
    assert _keep_coeff(S(1.0), x) == 1.0 * x
    assert _keep_coeff(S(-1.0), x) == -1.0 * x
    assert _keep_coeff(S.One, 2 * x) == 2 * x
    assert _keep_coeff(S(2), x / 2) == x
    assert _keep_coeff(S(2), sin(x)) == 2 * sin(x)
    assert _keep_coeff(S(2), x + 1) == u
    assert _keep_coeff(x, 1 / x) == 1
    assert _keep_coeff(x + 1, S(2)) == u
    assert _keep_coeff(S.Half, S.One) == S.Half
    p = Pow(2, 3, evaluate=False)
    assert _keep_coeff(S(-1), p) == Mul(-1, p, evaluate=False)
    a = Add(2, p, evaluate=False)
    assert _keep_coeff(S.Half, a, clear=True) == Mul(S.Half, a, evaluate=False)
    assert _keep_coeff(S.Half, a, clear=False) == Add(1, Mul(S.Half, p, evaluate=False), evaluate=False)