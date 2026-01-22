from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_coset_table():
    G = PermutationGroup(Permutation(0, 1, 2, 3), Permutation(0, 1, 2), Permutation(0, 4, 2, 7), Permutation(5, 6), Permutation(0, 7))
    H = PermutationGroup(Permutation(0, 1, 2, 3), Permutation(0, 7))
    assert G.coset_table(H) == [[0, 0, 0, 0, 1, 2, 3, 3, 0, 0], [4, 5, 2, 5, 6, 0, 7, 7, 1, 1], [5, 4, 5, 1, 0, 6, 8, 8, 6, 6], [3, 3, 3, 3, 7, 8, 0, 0, 3, 3], [2, 1, 4, 4, 4, 4, 9, 9, 4, 4], [1, 2, 1, 2, 5, 5, 10, 10, 5, 5], [6, 6, 6, 6, 2, 1, 11, 11, 2, 2], [9, 10, 8, 10, 11, 3, 1, 1, 7, 7], [10, 9, 10, 7, 3, 11, 2, 2, 11, 11], [8, 7, 9, 9, 9, 9, 4, 4, 9, 9], [7, 8, 7, 8, 10, 10, 5, 5, 10, 10], [11, 11, 11, 11, 8, 7, 6, 6, 8, 8]]