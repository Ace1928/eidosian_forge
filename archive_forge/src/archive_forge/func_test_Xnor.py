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
def test_Xnor():
    assert Xnor() is true
    assert Xnor(A) == ~A
    assert Xnor(A, A) is true
    assert Xnor(True, A, A) is false
    assert Xnor(A, A, A, A, A) == ~A
    assert Xnor(True) is false
    assert Xnor(False) is true
    assert Xnor(True, True) is true
    assert Xnor(True, False) is false
    assert Xnor(False, False) is true
    assert Xnor(True, A) == A
    assert Xnor(False, A) == ~A
    assert Xnor(True, False, False) is false
    assert Xnor(True, False, A) == A
    assert Xnor(False, False, A) == ~A