from sympy.combinatorics.named_groups import SymmetricGroup, DihedralGroup,\
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.util import _check_cycles_alt_sym, _strip,\
from sympy.combinatorics.testutil import _verify_bsgs
def test_distribute_gens_by_base():
    base = [0, 1, 2]
    gens = [Permutation([0, 1, 2, 3]), Permutation([0, 1, 3, 2]), Permutation([0, 2, 3, 1]), Permutation([3, 2, 1, 0])]
    assert _distribute_gens_by_base(base, gens) == [gens, [Permutation([0, 1, 2, 3]), Permutation([0, 1, 3, 2]), Permutation([0, 2, 3, 1])], [Permutation([0, 1, 2, 3]), Permutation([0, 1, 3, 2])]]