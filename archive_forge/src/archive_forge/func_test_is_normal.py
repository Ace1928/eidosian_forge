from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_is_normal():
    gens_s5 = [Permutation(p) for p in [[1, 2, 3, 4, 0], [2, 1, 4, 0, 3]]]
    G1 = PermutationGroup(gens_s5)
    assert G1.order() == 120
    gens_a5 = [Permutation(p) for p in [[1, 0, 3, 2, 4], [2, 1, 4, 3, 0]]]
    G2 = PermutationGroup(gens_a5)
    assert G2.order() == 60
    assert G2.is_normal(G1)
    gens3 = [Permutation(p) for p in [[2, 1, 3, 0, 4], [1, 2, 0, 3, 4]]]
    G3 = PermutationGroup(gens3)
    assert not G3.is_normal(G1)
    assert G3.order() == 12
    G4 = G1.normal_closure(G3.generators)
    assert G4.order() == 60
    gens5 = [Permutation(p) for p in [[1, 2, 3, 0, 4], [1, 2, 0, 3, 4]]]
    G5 = PermutationGroup(gens5)
    assert G5.order() == 24
    G6 = G1.normal_closure(G5.generators)
    assert G6.order() == 120
    assert G1.is_subgroup(G6)
    assert not G1.is_subgroup(G4)
    assert G2.is_subgroup(G4)
    I5 = PermutationGroup(Permutation(4))
    assert I5.is_normal(G5)
    assert I5.is_normal(G6, strict=False)
    p1 = Permutation([1, 0, 2, 3, 4])
    p2 = Permutation([0, 1, 2, 4, 3])
    p3 = Permutation([3, 4, 2, 1, 0])
    id_ = Permutation([0, 1, 2, 3, 4])
    H = PermutationGroup([p1, p3])
    H_n1 = PermutationGroup([p1, p2])
    H_n2_1 = PermutationGroup(p1)
    H_n2_2 = PermutationGroup(p2)
    H_id = PermutationGroup(id_)
    assert H_n1.is_normal(H)
    assert H_n2_1.is_normal(H_n1)
    assert H_n2_2.is_normal(H_n1)
    assert H_id.is_normal(H_n2_1)
    assert H_id.is_normal(H_n1)
    assert H_id.is_normal(H)
    assert not H_n2_1.is_normal(H)
    assert not H_n2_2.is_normal(H)