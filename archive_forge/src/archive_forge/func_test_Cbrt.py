from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.codegen.cfunctions import (
from sympy.core.function import expand_log
def test_Cbrt():
    x = Symbol('x')
    assert Cbrt(x).expand(func=True) - x ** Rational(1, 3) == 0
    assert Cbrt(42 * x).diff(x) - 42 * (42 * x) ** (Rational(1, 3) - 1) / 3 == 0
    assert Cbrt(42 * x).diff(x) - Cbrt(42 * x).expand(func=True).diff(x) == 0