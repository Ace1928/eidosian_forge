from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.numbers import oo
from sympy.core.relational import Equality, Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.sets.sets import (Interval, Union)
from sympy.simplify.simplify import simplify
from sympy.logic.boolalg import (
from sympy.assumptions.cnf import CNF
from sympy.testing.pytest import raises, XFAIL, slow
from itertools import combinations, permutations, product
@slow
def test_relational_simplification_numerically():

    def test_simplification_numerically_function(original, simplified):
        symb = original.free_symbols
        n = len(symb)
        valuelist = list(set(combinations(list(range(-(n - 1), n)) * n, n)))
        for values in valuelist:
            sublist = dict(zip(symb, values))
            originalvalue = original.subs(sublist)
            simplifiedvalue = simplified.subs(sublist)
            assert originalvalue == simplifiedvalue, 'Original: {}\nand simplified: {}\ndo not evaluate to the same value for {}'.format(original, simplified, sublist)
    w, x, y, z = symbols('w x y z', real=True)
    d, e = symbols('d e', real=False)
    expressions = (And(Eq(x, y), x >= y, w < y, y >= z, z < y), And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y), Or(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y), And(x >= y, Eq(y, x)), Or(And(Eq(x, y), x >= y, w < y, Or(y >= z, z < y)), And(Eq(x, y), x >= 1, 2 < y, y >= -1, z < y)), Eq(x, y) & Eq(d, e) & (x >= y) & (d >= e))
    for expression in expressions:
        test_simplification_numerically_function(expression, expression.simplify())