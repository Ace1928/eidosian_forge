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
def test_subs_constants():
    a, b = symbols('a b', commutative=True)
    x, y = symbols('x y', commutative=False)
    assert (a * b).subs(2 * a, 1) == a * b
    assert (1.5 * a * b).subs(a, 1) == 1.5 * b
    assert (2 * a * b).subs(2 * a, 1) == b
    assert (2 * a * b).subs(4 * a, 1) == 2 * a * b
    assert (x * y).subs(2 * x, 1) == x * y
    assert (1.5 * x * y).subs(x, 1) == 1.5 * y
    assert (2 * x * y).subs(2 * x, 1) == y
    assert (2 * x * y).subs(4 * x, 1) == 2 * x * y