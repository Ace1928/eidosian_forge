from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_abelian_invariants():
    G = AbelianGroup(2, 3, 4)
    assert G.abelian_invariants() == [2, 3, 4]
    G = PermutationGroup([Permutation(1, 2, 3, 4), Permutation(1, 2), Permutation(5, 6)])
    assert G.abelian_invariants() == [2, 2]
    G = AlternatingGroup(7)
    assert G.abelian_invariants() == []
    G = AlternatingGroup(4)
    assert G.abelian_invariants() == [3]
    G = DihedralGroup(4)
    assert G.abelian_invariants() == [2, 2]
    G = PermutationGroup([Permutation(1, 2, 3, 4, 5, 6, 7)])
    assert G.abelian_invariants() == [7]
    G = DihedralGroup(12)
    S = G.sylow_subgroup(3)
    assert S.abelian_invariants() == [3]
    G = PermutationGroup(Permutation(0, 1, 2), Permutation(0, 2, 3))
    assert G.abelian_invariants() == [3]
    G = PermutationGroup([Permutation(0, 1), Permutation(0, 2, 4, 6)(1, 3, 5, 7)])
    assert G.abelian_invariants() == [2, 4]
    G = SymmetricGroup(30)
    S = G.sylow_subgroup(2)
    assert S.abelian_invariants() == [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    S = G.sylow_subgroup(3)
    assert S.abelian_invariants() == [3, 3, 3, 3]
    S = G.sylow_subgroup(5)
    assert S.abelian_invariants() == [5, 5, 5]