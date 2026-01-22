from sympy.combinatorics.named_groups import (SymmetricGroup, CyclicGroup,
from sympy.testing.pytest import raises
def test_SymmetricGroup():
    G = SymmetricGroup(5)
    elements = list(G.generate())
    assert G.generators[0].size == 5
    assert len(elements) == 120
    assert G.is_solvable is False
    assert G.is_abelian is False
    assert G.is_nilpotent is False
    assert G.is_transitive() is True
    H = SymmetricGroup(1)
    assert H.order() == 1
    L = SymmetricGroup(2)
    assert L.order() == 2