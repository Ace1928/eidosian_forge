from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import (Derivative, Function, Lambda, Subs)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.core.sympify import SympifyError
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, cot, sin, tan)
from sympy.matrices.dense import (Matrix, zeros)
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import RootOf
from sympy.simplify.cse_main import cse
from sympy.simplify.simplify import nsimplify
from sympy.core.basic import _aresame
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import a, x, y, z, t
def test_derivative_subs2():
    f_func, g_func = symbols('f g', cls=Function)
    f, g = (f_func(x, y, z), g_func(x, y, z))
    assert Derivative(f, x, y).subs(Derivative(f, x, y), g) == g
    assert Derivative(f, y, x).subs(Derivative(f, x, y), g) == g
    assert Derivative(f, x, y).subs(Derivative(f, x), g) == Derivative(g, y)
    assert Derivative(f, x, y).subs(Derivative(f, y), g) == Derivative(g, x)
    assert Derivative(f, x, y, z).subs(Derivative(f, x, z), g) == Derivative(g, y)
    assert Derivative(f, x, y, z).subs(Derivative(f, z, y), g) == Derivative(g, x)
    assert Derivative(f, x, y, z).subs(Derivative(f, z, y, x), g) == g
    assert Derivative(f, x, x, y).subs(Derivative(f, y, y), g) == Derivative(f, x, x, y)
    assert Derivative(f, x, y, y, z).subs(Derivative(f, x, y, y, y), g) == Derivative(f, x, y, y, z)
    assert Derivative(f, x, y).subs(Derivative(f_func(x), x, y), g) == Derivative(f, x, y)