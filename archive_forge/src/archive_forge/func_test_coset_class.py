from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_coset_class():
    a = Permutation(1, 2)
    b = Permutation(0, 1)
    G = PermutationGroup([a, b])
    rht_coset = G * a
    assert rht_coset.is_right_coset
    assert not rht_coset.is_left_coset
    list_repr = rht_coset.as_list()
    expected = [Permutation(0, 2), Permutation(0, 2, 1), Permutation(1, 2), Permutation(2), Permutation(2)(0, 1), Permutation(0, 1, 2)]
    for ele in list_repr:
        assert ele in expected
    left_coset = a * G
    assert not left_coset.is_right_coset
    assert left_coset.is_left_coset
    list_repr = left_coset.as_list()
    expected = [Permutation(2)(0, 1), Permutation(0, 1, 2), Permutation(1, 2), Permutation(2), Permutation(0, 2), Permutation(0, 2, 1)]
    for ele in list_repr:
        assert ele in expected
    G = PermutationGroup(Permutation(1, 2, 3, 4), Permutation(2, 3, 4))
    H = PermutationGroup(Permutation(1, 2, 3, 4))
    g = Permutation(1, 3)(2, 4)
    rht_coset = Coset(g, H, G, dir='+')
    assert rht_coset.is_right_coset
    list_repr = rht_coset.as_list()
    expected = [Permutation(1, 2, 3, 4), Permutation(4), Permutation(1, 3)(2, 4), Permutation(1, 4, 3, 2)]
    for ele in list_repr:
        assert ele in expected