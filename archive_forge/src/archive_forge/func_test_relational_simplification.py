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
def test_relational_simplification():
    w, x, y, z = symbols('w x y z', real=True)
    d, e = symbols('d e', real=False)
    assert Or(x >= y, x < y).simplify() == S.true
    assert Or(x >= y, y > x).simplify() == S.true
    assert Or(x >= y, -x > -y).simplify() == S.true
    assert Or(x >= y, -y < -x).simplify() == S.true
    assert Or(-x <= -y, x < y).simplify() == S.true
    assert Or(-x <= -y, -x > -y).simplify() == S.true
    assert Or(-x <= -y, y > x).simplify() == S.true
    assert Or(-x <= -y, -y < -x).simplify() == S.true
    assert Or(y <= x, x < y).simplify() == S.true
    assert Or(y <= x, y > x).simplify() == S.true
    assert Or(y <= x, -x > -y).simplify() == S.true
    assert Or(y <= x, -y < -x).simplify() == S.true
    assert Or(-y >= -x, x < y).simplify() == S.true
    assert Or(-y >= -x, y > x).simplify() == S.true
    assert Or(-y >= -x, -x > -y).simplify() == S.true
    assert Or(-y >= -x, -y < -x).simplify() == S.true
    assert Or(x < y, x >= y).simplify() == S.true
    assert Or(y > x, x >= y).simplify() == S.true
    assert Or(-x > -y, x >= y).simplify() == S.true
    assert Or(-y < -x, x >= y).simplify() == S.true
    assert Or(x < y, -x <= -y).simplify() == S.true
    assert Or(-x > -y, -x <= -y).simplify() == S.true
    assert Or(y > x, -x <= -y).simplify() == S.true
    assert Or(-y < -x, -x <= -y).simplify() == S.true
    assert Or(x < y, y <= x).simplify() == S.true
    assert Or(y > x, y <= x).simplify() == S.true
    assert Or(-x > -y, y <= x).simplify() == S.true
    assert Or(-y < -x, y <= x).simplify() == S.true
    assert Or(x < y, -y >= -x).simplify() == S.true
    assert Or(y > x, -y >= -x).simplify() == S.true
    assert Or(-x > -y, -y >= -x).simplify() == S.true
    assert Or(-y < -x, -y >= -x).simplify() == S.true
    assert Or(x >= y, w < z, x <= y).simplify() == S.true
    assert And(x >= y, x < y).simplify() == S.false
    assert Or(x >= y, Eq(y, x)).simplify() == (x >= y)
    assert And(x >= y, Eq(y, x)).simplify() == Eq(x, y)
    assert And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y).simplify() == Eq(x, y) & (x >= 1) & (y >= 5) & (y > z)
    assert Or(Eq(x, y), x >= y, w < y, z < y).simplify() == (x >= y) | (y > z) | (w < y)
    assert And(Eq(x, y), x >= y, w < y, y >= z, z < y).simplify() == Eq(x, y) & (y > z) & (w < y)
    assert And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y).simplify() == Eq(x, y) & (x >= 1) & (y >= 5) & (y > z)
    assert (Eq(x, y) & Eq(d, e) & (x >= y) & (d >= e)).simplify() == Eq(x, y) & Eq(d, e) & (d >= e)
    assert And(Eq(x, y), Eq(x, -y)).simplify() == And(Eq(x, 0), Eq(y, 0))
    assert Xor(x >= y, x <= y).simplify() == Ne(x, y)
    assert And(x > 1, x < -1, Eq(x, y)).simplify() == S.false
    assert And(x >= y, Eq(y, 0)).simplify() == And(x >= 0, Eq(y, 0))
    assert Or(Ne(x, 1), Ne(x, 2)).simplify() == S.true
    assert And(Eq(x, 1), Ne(2, x)).simplify() == Eq(x, 1)
    assert Or(Eq(x, 1), Ne(2, x)).simplify() == Ne(x, 2)