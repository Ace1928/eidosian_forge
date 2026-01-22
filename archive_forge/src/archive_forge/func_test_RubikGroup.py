from sympy.combinatorics.named_groups import (SymmetricGroup, CyclicGroup,
from sympy.testing.pytest import raises
def test_RubikGroup():
    raises(ValueError, lambda: RubikGroup(1))