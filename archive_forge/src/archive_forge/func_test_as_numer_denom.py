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
def test_as_numer_denom():
    a, b, c = symbols('a, b, c')
    assert nan.as_numer_denom() == (nan, 1)
    assert oo.as_numer_denom() == (oo, 1)
    assert (-oo).as_numer_denom() == (-oo, 1)
    assert zoo.as_numer_denom() == (zoo, 1)
    assert (-zoo).as_numer_denom() == (zoo, 1)
    assert x.as_numer_denom() == (x, 1)
    assert (1 / x).as_numer_denom() == (1, x)
    assert (x / y).as_numer_denom() == (x, y)
    assert (x / 2).as_numer_denom() == (x, 2)
    assert (x * y / z).as_numer_denom() == (x * y, z)
    assert (x / (y * z)).as_numer_denom() == (x, y * z)
    assert S.Half.as_numer_denom() == (1, 2)
    assert (1 / y ** 2).as_numer_denom() == (1, y ** 2)
    assert (x / y ** 2).as_numer_denom() == (x, y ** 2)
    assert ((x ** 2 + 1) / y).as_numer_denom() == (x ** 2 + 1, y)
    assert (x * (y + 1) / y ** 7).as_numer_denom() == (x * (y + 1), y ** 7)
    assert (x ** (-2)).as_numer_denom() == (1, x ** 2)
    assert (a / x + b / 2 / x + c / 3 / x).as_numer_denom() == (6 * a + 3 * b + 2 * c, 6 * x)
    assert (a / x + b / 2 / x + c / 3 / y).as_numer_denom() == (2 * c * x + y * (6 * a + 3 * b), 6 * x * y)
    assert (a / x + b / 2 / x + c / 0.5 / x).as_numer_denom() == (2 * a + b + 4.0 * c, 2 * x)
    assert int(log(Add(*[Dummy() / i / x for i in range(1, 705)]).as_numer_denom()[1] / x).n(4)) == 705
    for i in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
        assert (i + x / 3).as_numer_denom() == (x + i, 3)
    assert (S.Infinity + x / 3 + y / 4).as_numer_denom() == (4 * x + 3 * y + S.Infinity, 12)
    assert (oo * x + zoo * y).as_numer_denom() == (zoo * y + oo * x, 1)
    A, B, C = symbols('A,B,C', commutative=False)
    assert (A * B * C ** (-1)).as_numer_denom() == (A * B * C ** (-1), 1)
    assert (A * B * C ** (-1) / x).as_numer_denom() == (A * B * C ** (-1), x)
    assert (C ** (-1) * A * B).as_numer_denom() == (C ** (-1) * A * B, 1)
    assert (C ** (-1) * A * B / x).as_numer_denom() == (C ** (-1) * A * B, x)
    assert ((A * B * C) ** (-1)).as_numer_denom() == ((A * B * C) ** (-1), 1)
    assert ((A * B * C) ** (-1) / x).as_numer_denom() == ((A * B * C) ** (-1), x)
    assert Add(0, (x + y) / z / -2, evaluate=False).as_numer_denom() == (-x - y, 2 * z)