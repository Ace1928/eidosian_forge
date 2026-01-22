from sympy.assumptions.refine import refine
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import (ExprBuilder, unchanged, Expr,
from sympy.core.function import (Function, expand, WildFunction,
from sympy.core.mul import Mul
from sympy.core.numbers import (NumberSymbol, E, zoo, oo, Float, I,
from sympy.core.power import Pow
from sympy.core.relational import Ge, Lt, Gt, Le
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols, Dummy, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import tan, sin, cos
from sympy.functions.special.delta_functions import (Heaviside,
from sympy.functions.special.error_functions import Si
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate, Integral
from sympy.physics.secondquant import FockState
from sympy.polys.partfrac import apart
from sympy.polys.polytools import factor, cancel, Poly
from sympy.polys.rationaltools import together
from sympy.series.order import O
from sympy.sets.sets import FiniteSet
from sympy.simplify.combsimp import combsimp
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import collect, radsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify, nsimplify
from sympy.simplify.trigsimp import trigsimp
from sympy.tensor.indexed import Indexed
from sympy.physics.units import meter
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import a, b, c, n, t, u, x, y, z
def test_is_meromorphic():
    f = a / x ** 2 + b + x + c * x ** 2
    assert f.is_meromorphic(x, 0) is True
    assert f.is_meromorphic(x, 1) is True
    assert f.is_meromorphic(x, zoo) is True
    g = 3 + 2 * x ** (log(3) / log(2) - 1)
    assert g.is_meromorphic(x, 0) is False
    assert g.is_meromorphic(x, 1) is True
    assert g.is_meromorphic(x, zoo) is False
    n = Symbol('n', integer=True)
    e = sin(1 / x) ** n * x
    assert e.is_meromorphic(x, 0) is False
    assert e.is_meromorphic(x, 1) is True
    assert e.is_meromorphic(x, zoo) is False
    e = log(x) ** pi
    assert e.is_meromorphic(x, 0) is False
    assert e.is_meromorphic(x, 1) is False
    assert e.is_meromorphic(x, 2) is True
    assert e.is_meromorphic(x, zoo) is False
    assert (log(x) ** a).is_meromorphic(x, 0) is False
    assert (log(x) ** a).is_meromorphic(x, 1) is False
    assert (a ** log(x)).is_meromorphic(x, 0) is None
    assert (3 ** log(x)).is_meromorphic(x, 0) is False
    assert (3 ** log(x)).is_meromorphic(x, 1) is True