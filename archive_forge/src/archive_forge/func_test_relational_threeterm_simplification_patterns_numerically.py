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
def test_relational_threeterm_simplification_patterns_numerically():
    from sympy.core import Wild
    from sympy.logic.boolalg import _simplify_patterns_and3
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    symb = [a, b, c]
    patternlists = [[And, _simplify_patterns_and3()]]
    valuelist = list(set(combinations(list(range(-2, 3)) * 3, 3)))
    valuelist = [v for v in valuelist if any([w % 2 for w in v]) or not any(v)]
    for func, patternlist in patternlists:
        for pattern in patternlist:
            original = func(*pattern[0].args)
            simplified = pattern[1]
            for values in valuelist:
                sublist = dict(zip(symb, values))
                originalvalue = original.xreplace(sublist)
                simplifiedvalue = simplified.xreplace(sublist)
                assert originalvalue == simplifiedvalue, 'Original: {}\nand simplified: {}\ndo not evaluate to the same value for{}'.format(pattern[0], simplified, sublist)