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
def test_issue_15084_13166():
    eq = f(x, g(x))
    assert eq.diff((g(x), y)) == Derivative(f(x, g(x)), (g(x), y))
    assert eq.diff(x, 2).doit() == (Derivative(f(x, g(x)), (g(x), 2)) * Derivative(g(x), x) + Subs(Derivative(f(x, _xi_2), _xi_2, x), _xi_2, g(x))) * Derivative(g(x), x) + Derivative(f(x, g(x)), g(x)) * Derivative(g(x), (x, 2)) + Derivative(g(x), x) * Subs(Derivative(f(_xi_1, g(x)), _xi_1, g(x)), _xi_1, x) + Subs(Derivative(f(_xi_1, g(x)), (_xi_1, 2)), _xi_1, x)
    assert diff(f(x, t, g(x, t)), x).doit() == Derivative(f(x, t, g(x, t)), g(x, t)) * Derivative(g(x, t), x) + Subs(Derivative(f(_xi_1, t, g(x, t)), _xi_1), _xi_1, x)
    assert eq.diff(x, g(x)) == eq.diff(g(x), x)