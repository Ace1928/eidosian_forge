from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroup__getitem__():
    assert F[0:] == FreeGroup('x, y, z')
    assert F[1:] == FreeGroup('y, z')
    assert F[2:] == FreeGroup('z')