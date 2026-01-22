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
def test_factor_large():
    f = (x ** 2 + 4 * x + 4) ** 10000000 * (x ** 2 + 1) * (x ** 2 + 2 * x + 1) ** 1234567
    g = (x ** 2 + 2 * x + 1) ** 3000 * y ** 2 + (x ** 2 + 2 * x + 1) ** 3000 * 2 * y + (x ** 2 + 2 * x + 1) ** 3000
    assert factor(f) == (x + 2) ** 20000000 * (x ** 2 + 1) * (x + 1) ** 2469134
    assert factor(g) == (x + 1) ** 6000 * (y + 1) ** 2
    assert factor_list(f) == (1, [(x + 1, 2469134), (x + 2, 20000000), (x ** 2 + 1, 1)])
    assert factor_list(g) == (1, [(y + 1, 2), (x + 1, 6000)])
    f = (x ** 2 - y ** 2) ** 200000 * (x ** 7 + 1)
    g = (x ** 2 + y ** 2) ** 200000 * (x ** 7 + 1)
    assert factor(f) == (x + 1) * (x - y) ** 200000 * (x + y) ** 200000 * (x ** 6 - x ** 5 + x ** 4 - x ** 3 + x ** 2 - x + 1)
    assert factor(g, gaussian=True) == (x + 1) * (x - I * y) ** 200000 * (x + I * y) ** 200000 * (x ** 6 - x ** 5 + x ** 4 - x ** 3 + x ** 2 - x + 1)
    assert factor_list(f) == (1, [(x + 1, 1), (x - y, 200000), (x + y, 200000), (x ** 6 - x ** 5 + x ** 4 - x ** 3 + x ** 2 - x + 1, 1)])
    assert factor_list(g, gaussian=True) == (1, [(x + 1, 1), (x - I * y, 200000), (x + I * y, 200000), (x ** 6 - x ** 5 + x ** 4 - x ** 3 + x ** 2 - x + 1, 1)])