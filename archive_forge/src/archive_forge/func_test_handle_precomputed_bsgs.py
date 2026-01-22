from sympy.combinatorics.named_groups import SymmetricGroup, DihedralGroup,\
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.util import _check_cycles_alt_sym, _strip,\
from sympy.combinatorics.testutil import _verify_bsgs
def test_handle_precomputed_bsgs():
    A = AlternatingGroup(5)
    A.schreier_sims()
    base = A.base
    strong_gens = A.strong_gens
    result = _handle_precomputed_bsgs(base, strong_gens)
    strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    assert strong_gens_distr == result[2]
    transversals = result[0]
    orbits = result[1]
    base_len = len(base)
    for i in range(base_len):
        for el in orbits[i]:
            assert transversals[i][el](base[i]) == el
            for j in range(i):
                assert transversals[i][el](base[j]) == base[j]
    order = 1
    for i in range(base_len):
        order *= len(orbits[i])
    assert A.order() == order