import sympy
from sympy.parsing.sympy_parser import (
from sympy.testing.pytest import raises
def test_function_exponentiation():
    cases = {'sin**2(x)': 'sin(x)**2', 'exp^y(z)': 'exp(z)^y', 'sin**2(E^(x))': 'sin(E^(x))**2'}
    transformations = standard_transformations + (convert_xor,)
    transformations2 = transformations + (function_exponentiation,)
    for case in cases:
        implicit = parse_expr(case, transformations=transformations2)
        normal = parse_expr(cases[case], transformations=transformations)
        assert implicit == normal
    other_implicit = ['x y', 'x sin x', '2x', 'sin x', 'cos 2*x', 'sin cos x']
    for case in other_implicit:
        raises(SyntaxError, lambda: parse_expr(case, transformations=transformations2))
    assert parse_expr('x**2', local_dict={'x': sympy.Symbol('x')}, transformations=transformations2) == parse_expr('x**2')