from sympy.core import symbols, Symbol, Tuple, oo, Dummy
from sympy.tensor.indexed import IndexException
from sympy.testing.pytest import raises
from sympy.utilities.iterables import iterable
from sympy.concrete.summations import Sum
from sympy.core.function import Function, Subs, Derivative
from sympy.core.relational import (StrictLessThan, GreaterThan,
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.series.order import Order
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import IndexedBase, Idx, Indexed
def test_complicated_derivative_with_Indexed():
    x, y = symbols('x,y', cls=IndexedBase)
    sigma = symbols('sigma')
    i, j, k = symbols('i,j,k')
    m0, m1, m2, m3, m4, m5 = symbols('m0:6')
    f = Function('f')
    expr = f((x[i] - y[i]) ** 2 / sigma)
    _xi_1 = symbols('xi_1', cls=Dummy)
    assert expr.diff(x[m0]).dummy_eq((x[i] - y[i]) * KroneckerDelta(i, m0) * 2 * Subs(Derivative(f(_xi_1), _xi_1), (_xi_1,), ((x[i] - y[i]) ** 2 / sigma,)) / sigma)
    assert expr.diff(x[m0]).diff(x[m1]).dummy_eq(2 * KroneckerDelta(i, m0) * KroneckerDelta(i, m1) * Subs(Derivative(f(_xi_1), _xi_1), (_xi_1,), ((x[i] - y[i]) ** 2 / sigma,)) / sigma + 4 * (x[i] - y[i]) ** 2 * KroneckerDelta(i, m0) * KroneckerDelta(i, m1) * Subs(Derivative(f(_xi_1), _xi_1, _xi_1), (_xi_1,), ((x[i] - y[i]) ** 2 / sigma,)) / sigma ** 2)