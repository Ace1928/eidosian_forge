from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_orbit_rep():
    G = DihedralGroup(6)
    assert G.orbit_rep(1, 3) in [Permutation([2, 3, 4, 5, 0, 1]), Permutation([4, 3, 2, 1, 0, 5])]
    H = CyclicGroup(4) * G
    assert H.orbit_rep(1, 5) is False