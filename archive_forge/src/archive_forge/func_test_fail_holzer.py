from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.ntheory.factor_ import factorint
from sympy.simplify.powsimp import powsimp
from sympy.core.function import _mexpand
from sympy.core.sorting import default_sort_key, ordered
from sympy.functions.elementary.trigonometric import sin
from sympy.solvers.diophantine import diophantine
from sympy.solvers.diophantine.diophantine import (diop_DN,
from sympy.testing.pytest import slow, raises, XFAIL
from sympy.utilities.iterables import (
@XFAIL
def test_fail_holzer():
    eq = lambda x, y, z: a * x ** 2 + b * y ** 2 - c * z ** 2
    a, b, c = (4, 79, 23)
    x, y, z = xyz = (26, 1, 11)
    X, Y, Z = ans = (2, 7, 13)
    assert eq(*xyz) == 0
    assert eq(*ans) == 0
    assert max(a * x ** 2, b * y ** 2, c * z ** 2) <= a * b * c
    assert max(a * X ** 2, b * Y ** 2, c * Z ** 2) <= a * b * c
    h = holzer(x, y, z, a, b, c)
    assert h == ans