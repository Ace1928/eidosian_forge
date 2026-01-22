from sympy.concrete.summations import Sum
from sympy.core.basic import Basic, _aresame
from sympy.core.cache import clear_cache
from sympy.core.containers import Dict, Tuple
from sympy.core.expr import Expr, unchanged
from sympy.core.function import (Subs, Function, diff, Lambda, expand,
from sympy.core.numbers import E, Float, zoo, Rational, pi, I, oo, nan
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Dummy, Symbol
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin, cos, acos
from sympy.functions.special.error_functions import expint
from sympy.functions.special.gamma_functions import loggamma, polygamma
from sympy.matrices.dense import Matrix
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.tensor.indexed import Indexed
from sympy.core.function import (PoleError, _mexpand, arity,
from sympy.core.parameters import _exp_is_pow
from sympy.core.sympify import sympify, SympifyError
from sympy.matrices import MutableMatrix, ImmutableMatrix
from sympy.sets.sets import FiniteSet
from sympy.solvers.solveset import solveset
from sympy.tensor.array import NDimArray
from sympy.utilities.iterables import subsets, variations
from sympy.testing.pytest import XFAIL, raises, warns_deprecated_sympy, _both_exp_pow
from sympy.abc import t, w, x, y, z
def test_deriv1():
    assert f(2 * x).diff(x) == 2 * Subs(Derivative(f(x), x), x, 2 * x)
    assert (f(x) ** 3).diff(x) == 3 * f(x) ** 2 * f(x).diff(x)
    assert (f(2 * x) ** 3).diff(x) == 6 * f(2 * x) ** 2 * Subs(Derivative(f(x), x), x, 2 * x)
    assert f(2 + x).diff(x) == Subs(Derivative(f(x), x), x, x + 2)
    assert f(2 + 3 * x).diff(x) == 3 * Subs(Derivative(f(x), x), x, 3 * x + 2)
    assert f(3 * sin(x)).diff(x) == 3 * cos(x) * Subs(Derivative(f(x), x), x, 3 * sin(x))
    assert f(x, x + z).diff(x) == Subs(Derivative(f(y, x + z), y), y, x) + Subs(Derivative(f(x, y), y), y, x + z)
    assert f(x, x ** 2).diff(x) == 2 * x * Subs(Derivative(f(x, y), y), y, x ** 2) + Subs(Derivative(f(y, x ** 2), y), y, x)
    assert f(x, g(y)).diff(g(y)) == Derivative(f(x, g(y)), g(y))