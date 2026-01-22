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
def test_subs_in_derivative():
    expr = sin(x * exp(y))
    u = Function('u')
    v = Function('v')
    assert Derivative(expr, y).subs(expr, y) == Derivative(y, y)
    assert Derivative(expr, y).subs(y, x).doit() == Derivative(expr, y).doit().subs(y, x)
    assert Derivative(f(x, y), y).subs(y, x) == Subs(Derivative(f(x, y), y), y, x)
    assert Derivative(f(x, y), y).subs(x, y) == Subs(Derivative(f(x, y), y), x, y)
    assert Derivative(f(x, y), y).subs(y, g(x, y)) == Subs(Derivative(f(x, y), y), y, g(x, y)).doit()
    assert Derivative(f(x, y), y).subs(x, g(x, y)) == Subs(Derivative(f(x, y), y), x, g(x, y))
    assert Derivative(f(x, y), g(y)).subs(x, g(x, y)) == Derivative(f(g(x, y), y), g(y))
    assert Derivative(f(u(x), h(y)), h(y)).subs(h(y), g(x, y)) == Subs(Derivative(f(u(x), h(y)), h(y)), h(y), g(x, y)).doit()
    assert Derivative(f(x, y), y).subs(y, z) == Derivative(f(x, z), z)
    assert Derivative(f(x, y), y).subs(y, g(y)) == Derivative(f(x, g(y)), g(y))
    assert Derivative(f(g(x), h(y)), h(y)).subs(h(y), u(y)) == Derivative(f(g(x), u(y)), u(y))
    assert Derivative(f(x, f(x, x)), f(x, x)).subs(f, Lambda((x, y), x + y)) == Subs(Derivative(z + x, z), z, 2 * x)
    assert Subs(Derivative(f(f(x)), x), f, cos).doit() == sin(x) * sin(cos(x))
    assert Subs(Derivative(f(f(x)), f(x)), f, cos).doit() == -sin(cos(x))
    assert isinstance(v(x, y, u(x, y)).diff(y).diff(x).diff(y), Expr)
    F = Lambda((x, y), exp(2 * x + 3 * y))
    abstract = f(x, f(x, x)).diff(x, 2)
    concrete = F(x, F(x, x)).diff(x, 2)
    assert (abstract.subs(f, F).doit() - concrete).simplify() == 0
    assert x in f(x).diff(x).subs(x, 0).atoms()
    assert Derivative(f(x, f(x, y)), x, y).subs(x, g(y)) == Subs(Derivative(f(x, f(x, y)), x, y), x, g(y))
    assert Derivative(f(x, x), x).subs(x, 0) == Subs(Derivative(f(x, x), x), x, 0)
    assert Derivative(f(y, g(x)), (x, z)).subs(z, x) == Derivative(f(y, g(x)), (x, x))
    df = f(x).diff(x)
    assert df.subs(df, 1) is S.One
    assert df.diff(df) is S.One
    dxy = Derivative(f(x, y), x, y)
    dyx = Derivative(f(x, y), y, x)
    assert dxy.subs(Derivative(f(x, y), y, x), 1) is S.One
    assert dxy.diff(dyx) is S.One
    assert Derivative(f(x, y), x, 2, y, 3).subs(dyx, g(x, y)) == Derivative(g(x, y), x, 1, y, 2)
    assert Derivative(f(x, x - y), y).subs(x, x + y) == Subs(Derivative(f(x, x - y), y), x, x + y)