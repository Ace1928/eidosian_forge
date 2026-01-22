from sympy.multipledispatch.conflict import (supercedes, ordering, ambiguities,
def test_supercedes():
    assert supercedes([B], [A])
    assert supercedes([B, A], [A, A])
    assert not supercedes([B, A], [A, B])
    assert not supercedes([A], [B])