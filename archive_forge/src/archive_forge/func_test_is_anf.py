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
def test_is_anf():
    x, y = symbols('x,y')
    assert is_anf(true) is True
    assert is_anf(false) is True
    assert is_anf(x) is True
    assert is_anf(And(x, y)) is True
    assert is_anf(Xor(x, y, And(x, y))) is True
    assert is_anf(Xor(x, y, Or(x, y))) is False
    assert is_anf(Xor(Not(x), y)) is False