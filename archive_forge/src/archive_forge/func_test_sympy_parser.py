import sys
import builtins
import types
from sympy.assumptions import Q
from sympy.core import Symbol, Function, Float, Rational, Integer, I, Mul, Pow, Eq, Lt, Le, Gt, Ge, Ne
from sympy.functions import exp, factorial, factorial2, sin, Min, Max
from sympy.logic import And
from sympy.series import Limit
from sympy.testing.pytest import raises, skip
from sympy.parsing.sympy_parser import (
def test_sympy_parser():
    x = Symbol('x')
    inputs = {'2*x': 2 * x, '3.00': Float(3), '22/7': Rational(22, 7), '2+3j': 2 + 3 * I, 'exp(x)': exp(x), 'x!': factorial(x), 'x!!': factorial2(x), '(x + 1)! - 1': factorial(x + 1) - 1, '3.[3]': Rational(10, 3), '.0[3]': Rational(1, 30), '3.2[3]': Rational(97, 30), '1.3[12]': Rational(433, 330), '1 + 3.[3]': Rational(13, 3), '1 + .0[3]': Rational(31, 30), '1 + 3.2[3]': Rational(127, 30), '.[0011]': Rational(1, 909), '0.1[00102] + 1': Rational(366697, 333330), '1.[0191]': Rational(10190, 9999), '10!': 3628800, '-(2)': -Integer(2), '[-1, -2, 3]': [Integer(-1), Integer(-2), Integer(3)], 'Symbol("x").free_symbols': x.free_symbols, "S('S(3).n(n=3)')": Float(3, 3), 'factorint(12, visual=True)': Mul(Pow(2, 2, evaluate=False), Pow(3, 1, evaluate=False), evaluate=False), 'Limit(sin(x), x, 0, dir="-")': Limit(sin(x), x, 0, dir='-'), 'Q.even(x)': Q.even(x)}
    for text, result in inputs.items():
        assert parse_expr(text) == result
    raises(TypeError, lambda: parse_expr('x', standard_transformations))
    raises(TypeError, lambda: parse_expr('x', transformations=lambda x, y: 1))
    raises(TypeError, lambda: parse_expr('x', transformations=(lambda x, y: 1,)))
    raises(TypeError, lambda: parse_expr('x', transformations=((),)))
    raises(TypeError, lambda: parse_expr('x', {}, [], []))
    raises(TypeError, lambda: parse_expr('x', [], [], {}))
    raises(TypeError, lambda: parse_expr('x', [], [], {}))