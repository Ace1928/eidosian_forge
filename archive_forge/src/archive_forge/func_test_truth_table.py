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
def test_truth_table():
    assert list(truth_table(And(x, y), [x, y], input=False)) == [False, False, False, True]
    assert list(truth_table(x | y, [x, y], input=False)) == [False, True, True, True]
    assert list(truth_table(x >> y, [x, y], input=False)) == [True, True, False, True]
    assert list(truth_table(And(x, y), [x, y])) == [([0, 0], False), ([0, 1], False), ([1, 0], False), ([1, 1], True)]