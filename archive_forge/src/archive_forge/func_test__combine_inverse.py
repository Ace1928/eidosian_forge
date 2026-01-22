from sympy import abc
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import signsimp
from sympy.testing.pytest import XFAIL
def test__combine_inverse():
    x, y = symbols('x y')
    assert Mul._combine_inverse(x * I * y, x * I) == y
    assert Mul._combine_inverse(x * x ** (1 + y), x ** (1 + y)) == x
    assert Mul._combine_inverse(x * I * y, y * I) == x
    assert Mul._combine_inverse(oo * I * y, y * I) is oo
    assert Mul._combine_inverse(oo * I * y, oo * I) == y
    assert Mul._combine_inverse(oo * I * y, oo * I) == y
    assert Mul._combine_inverse(oo * y, -oo) == -y
    assert Mul._combine_inverse(-oo * y, oo) == -y
    assert Mul._combine_inverse(1 - exp(x / y), exp(x / y) - 1) == -1
    assert Add._combine_inverse(oo, oo) is S.Zero
    assert Add._combine_inverse(oo * I, oo * I) is S.Zero
    assert Add._combine_inverse(x * oo, x * oo) is S.Zero
    assert Add._combine_inverse(-x * oo, -x * oo) is S.Zero
    assert Add._combine_inverse((x - oo) * (x + oo), -oo)