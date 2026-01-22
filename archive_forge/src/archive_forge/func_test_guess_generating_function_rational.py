from sympy.concrete.guess import (
from sympy.concrete.products import Product
from sympy.core.function import Function
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.exponential import exp
def test_guess_generating_function_rational():
    x = Symbol('x')
    assert guess_generating_function_rational([fibonacci(k) for k in range(5, 15)]) == (3 * x + 5) / (-x ** 2 - x + 1)