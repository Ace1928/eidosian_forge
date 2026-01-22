from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import symbols
from sympy.core.singleton import S
from sympy.core.function import expand, Function
from sympy.core.numbers import I
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor
from sympy.core.traversal import preorder_traversal, use, postorder_traversal, iterargs, iterfreeargs
from sympy.functions.elementary.piecewise import ExprCondPair, Piecewise
from sympy.testing.pytest import warns_deprecated_sympy
from sympy.utilities.iterables import capture
def test_iterargs():
    f = Function('f')
    x = symbols('x')
    assert list(iterfreeargs(Integral(f(x), (f(x), 1)))) == [Integral(f(x), (f(x), 1)), 1]
    assert list(iterargs(Integral(f(x), (f(x), 1)))) == [Integral(f(x), (f(x), 1)), f(x), (f(x), 1), x, f(x), 1, x]