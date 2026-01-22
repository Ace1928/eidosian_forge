from sympy.physics.secondquant import (
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.repr import srepr
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import slow, raises
from sympy.printing.latex import latex
def test_index_permutations_with_dummies():
    a, b, c, d = symbols('a b c d')
    p, q, r, s = symbols('p q r s', cls=Dummy)
    f, g = map(Function, 'fg')
    P = PermutationOperator
    expr = f(a, b, p, q) - f(b, a, p, q)
    assert simplify_index_permutations(expr, [P(a, b)]) == P(a, b) * f(a, b, p, q)
    expected = P(a, b) * substitute_dummies(f(a, b, p, q))
    expr = f(a, b, p, q) - f(b, a, q, p)
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expected == substitute_dummies(result)
    expr = f(a, b, q, p) - f(b, a, p, q)
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expected == substitute_dummies(result)
    expr = f(a, b, q, p) - g(b, a, p, q)
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expr == result