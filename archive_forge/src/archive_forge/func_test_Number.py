from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import Derivative
from sympy.core.numbers import Integer, Rational, Float, oo
from sympy.core.relational import Rel
from sympy.core.symbol import symbols
from sympy.functions import sin
from sympy.integrals.integrals import Integral
from sympy.series.order import Order
from sympy.printing.precedence import precedence, PRECEDENCE
def test_Number():
    assert precedence(Integer(0)) == PRECEDENCE['Atom']
    assert precedence(Integer(1)) == PRECEDENCE['Atom']
    assert precedence(Integer(-1)) == PRECEDENCE['Add']
    assert precedence(Integer(10)) == PRECEDENCE['Atom']
    assert precedence(Rational(5, 2)) == PRECEDENCE['Mul']
    assert precedence(Rational(-5, 2)) == PRECEDENCE['Add']
    assert precedence(Float(5)) == PRECEDENCE['Atom']
    assert precedence(Float(-5)) == PRECEDENCE['Add']
    assert precedence(oo) == PRECEDENCE['Atom']
    assert precedence(-oo) == PRECEDENCE['Add']