from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.codegen.cfunctions import (
from sympy.core.function import expand_log
def test_fma():
    x, y, z = symbols('x y z')
    assert fma(x, y, z).expand(func=True) - x * y - z == 0
    expr = fma(17 * x, 42 * y, 101 * z)
    assert expr.diff(x) - expr.expand(func=True).diff(x) == 0
    assert expr.diff(y) - expr.expand(func=True).diff(y) == 0
    assert expr.diff(z) - expr.expand(func=True).diff(z) == 0
    assert expr.diff(x) - 17 * 42 * y == 0
    assert expr.diff(y) - 17 * 42 * x == 0
    assert expr.diff(z) - 101 == 0