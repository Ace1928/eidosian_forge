from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_orbits():
    a = Permutation([2, 0, 1])
    b = Permutation([2, 1, 0])
    g = PermutationGroup([a, b])
    assert g.orbit(0) == {0, 1, 2}
    assert g.orbits() == [{0, 1, 2}]
    assert g.is_transitive() and g.is_transitive(strict=False)
    assert g.orbit_transversal(0) == [Permutation([0, 1, 2]), Permutation([2, 0, 1]), Permutation([1, 2, 0])]
    assert g.orbit_transversal(0, True) == [(0, Permutation([0, 1, 2])), (2, Permutation([2, 0, 1])), (1, Permutation([1, 2, 0]))]
    G = DihedralGroup(6)
    transversal, slps = _orbit_transversal(G.degree, G.generators, 0, True, slp=True)
    for i, t in transversal:
        slp = slps[i]
        w = G.identity
        for s in slp:
            w = G.generators[s] * w
        assert w == t
    a = Permutation(list(range(1, 100)) + [0])
    G = PermutationGroup([a])
    assert [min(o) for o in G.orbits()] == [0]
    G = PermutationGroup(rubik_cube_generators())
    assert [min(o) for o in G.orbits()] == [0, 1]
    assert not G.is_transitive() and (not G.is_transitive(strict=False))
    G = PermutationGroup([Permutation(0, 1, 3), Permutation(3)(0, 1)])
    assert not G.is_transitive() and G.is_transitive(strict=False)
    assert PermutationGroup(Permutation(3)).is_transitive(strict=False) is False