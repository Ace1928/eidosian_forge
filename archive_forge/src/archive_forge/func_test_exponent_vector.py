from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup, DihedralGroup
from sympy.matrices import Matrix
def test_exponent_vector():
    Groups = [SymmetricGroup(3), SymmetricGroup(4), SymmetricGroup(9).sylow_subgroup(3), SymmetricGroup(9).sylow_subgroup(2), SymmetricGroup(8).sylow_subgroup(2)]
    for G in Groups:
        PcGroup = G.polycyclic_group()
        collector = PcGroup.collector
        pcgs = PcGroup.pcgs
        for gen in G.generators:
            exp = collector.exponent_vector(gen)
            g = Permutation()
            for i in range(len(exp)):
                g = g * pcgs[i] ** exp[i] if exp[i] else g
            assert g == gen