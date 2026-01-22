from sympy.combinatorics.named_groups import (SymmetricGroup, CyclicGroup,
from sympy.testing.pytest import raises
def test_AbelianGroup():
    A = AbelianGroup(3, 3, 3)
    assert A.order() == 27
    assert A.is_abelian is True