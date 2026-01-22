from sympy.combinatorics.named_groups import SymmetricGroup, DihedralGroup,\
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.util import _check_cycles_alt_sym, _strip,\
from sympy.combinatorics.testutil import _verify_bsgs
def test_base_ordering():
    base = [2, 4, 5]
    degree = 7
    assert _base_ordering(base, degree) == [3, 4, 0, 5, 1, 2, 6]