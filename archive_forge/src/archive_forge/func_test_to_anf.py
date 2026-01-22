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
def test_to_anf():
    x, y, z = symbols('x,y,z')
    assert to_anf(And(x, y)) == And(x, y)
    assert to_anf(Or(x, y)) == Xor(x, y, And(x, y))
    assert to_anf(Or(Implies(x, y), And(x, y), y)) == Xor(x, True, x & y, remove_true=False)
    assert to_anf(Or(Nand(x, y), Nor(x, y), Xnor(x, y), Implies(x, y))) == True
    assert to_anf(Or(x, Not(y), Nor(x, z), And(x, y), Nand(y, z))) == Xor(True, And(y, z), And(x, y, z), remove_true=False)
    assert to_anf(Xor(x, y)) == Xor(x, y)
    assert to_anf(Not(x)) == Xor(x, True, remove_true=False)
    assert to_anf(Nand(x, y)) == Xor(True, And(x, y), remove_true=False)
    assert to_anf(Nor(x, y)) == Xor(x, y, True, And(x, y), remove_true=False)
    assert to_anf(Implies(x, y)) == Xor(x, True, And(x, y), remove_true=False)
    assert to_anf(Equivalent(x, y)) == Xor(x, y, True, remove_true=False)
    assert to_anf(Nand(x | y, x >> y), deep=False) == Xor(True, And(Or(x, y), Implies(x, y)), remove_true=False)
    assert to_anf(Nor(x ^ y, x & y), deep=False) == Xor(True, Or(Xor(x, y), And(x, y)), remove_true=False)