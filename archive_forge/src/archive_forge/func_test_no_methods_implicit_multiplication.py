import sympy
from sympy.parsing.sympy_parser import (
from sympy.testing.pytest import raises
def test_no_methods_implicit_multiplication():
    u = sympy.Symbol('u')
    transformations = standard_transformations + (implicit_multiplication,)
    expr = parse_expr('x.is_polynomial(x)', transformations=transformations)
    assert expr == True
    expr = parse_expr('(exp(x) / (1 + exp(2x))).subs(exp(x), u)', transformations=transformations)
    assert expr == u / (u ** 2 + 1)