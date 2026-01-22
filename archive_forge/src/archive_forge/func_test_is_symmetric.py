from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_is_symmetric():
    a = Permutation(0, 1, 2)
    b = Permutation(0, 1, size=3)
    assert PermutationGroup(a, b).is_symmetric is True
    a = Permutation(0, 2, 1)
    b = Permutation(1, 2, size=3)
    assert PermutationGroup(a, b).is_symmetric is True
    a = Permutation(0, 1, 2, 3)
    b = Permutation(0, 3)(1, 2)
    assert PermutationGroup(a, b).is_symmetric is False