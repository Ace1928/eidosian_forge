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
def test_issue_6976():
    x, y = symbols('x y')
    assert (sqrt(x) ** 3 + sqrt(x) + x + x ** 2).subs(sqrt(x), y) == y ** 4 + y ** 3 + y ** 2 + y
    assert (x ** 4 + x ** 3 + x ** 2 + x + sqrt(x)).subs(x ** 2, y) == sqrt(x) + x ** 3 + x + y ** 2 + y
    assert x.subs(x ** 3, y) == x
    assert x.subs(x ** Rational(1, 3), y) == y ** 3
    x, y = symbols('x y', nonnegative=True)
    assert (x ** 4 + x ** 3 + x ** 2 + x + sqrt(x)).subs(x ** 2, y) == y ** Rational(1, 4) + y ** Rational(3, 2) + sqrt(y) + y ** 2 + y
    assert x.subs(x ** 3, y) == y ** Rational(1, 3)