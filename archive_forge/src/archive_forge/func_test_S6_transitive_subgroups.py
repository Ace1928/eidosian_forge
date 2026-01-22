from sympy.combinatorics.galois import (
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import (
def test_S6_transitive_subgroups():
    """
    Test enough characteristics to distinguish all 16 transitive subgroups.
    """
    ts = S6TransitiveSubgroups
    A6 = AlternatingGroup(6)
    for name, alt, order, is_isom, not_isom in [(ts.C6, False, 6, CyclicGroup(6), None), (ts.S3, False, 6, SymmetricGroup(3), None), (ts.D6, False, 12, None, None), (ts.A4, True, 12, None, None), (ts.G18, False, 18, None, None), (ts.A4xC2, False, 24, None, SymmetricGroup(4)), (ts.S4m, False, 24, SymmetricGroup(4), None), (ts.S4p, True, 24, None, None), (ts.G36m, False, 36, None, None), (ts.G36p, True, 36, None, None), (ts.S4xC2, False, 48, None, None), (ts.PSL2F5, True, 60, None, None), (ts.G72, False, 72, None, None), (ts.PGL2F5, False, 120, None, None), (ts.A6, True, 360, None, None), (ts.S6, False, 720, None, None)]:
        for G in get_versions_of_S6_subgroup(name):
            assert G.is_transitive()
            assert G.degree == 6
            assert G.is_subgroup(A6) is alt
            assert G.order() == order
            if is_isom:
                assert is_isomorphic(G, is_isom)
            if not_isom:
                assert not is_isomorphic(G, not_isom)