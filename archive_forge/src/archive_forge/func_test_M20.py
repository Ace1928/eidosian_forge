from sympy.combinatorics.galois import (
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import (
def test_M20():
    G = S5TransitiveSubgroups.M20.get_perm_group()
    S5 = SymmetricGroup(5)
    A5 = AlternatingGroup(5)
    assert G.is_subgroup(S5)
    assert not G.is_subgroup(A5)
    assert G.degree == 5
    assert G.is_transitive()
    assert G.order() == 20