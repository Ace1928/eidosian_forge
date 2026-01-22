from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_minimal_block():
    D = DihedralGroup(6)
    block_system = D.minimal_block([0, 3])
    for i in range(3):
        assert block_system[i] == block_system[i + 3]
    S = SymmetricGroup(6)
    assert S.minimal_block([0, 1]) == [0, 0, 0, 0, 0, 0]
    assert Tetra.pgroup.minimal_block([0, 1]) == [0, 0, 0, 0]
    P1 = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    P2 = PermutationGroup(Permutation(0, 1, 2, 3, 4, 5), Permutation(1, 5)(2, 4))
    assert P1.minimal_block([0, 2]) == [0, 1, 0, 1, 0, 1]
    assert P2.minimal_block([0, 2]) == [0, 1, 0, 1, 0, 1]