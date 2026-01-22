from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def bsgs_direct_product(base1, gens1, base2, gens2, signed=True):
    """
    Direct product of two BSGS.

    Parameters
    ==========

    base1 : base of the first BSGS.

    gens1 : strong generating sequence of the first BSGS.

    base2, gens2 : similarly for the second BSGS.

    signed : flag for signed permutations.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import (get_symmetric_group_sgs, bsgs_direct_product)
    >>> base1, gens1 = get_symmetric_group_sgs(1)
    >>> base2, gens2 = get_symmetric_group_sgs(2)
    >>> bsgs_direct_product(base1, gens1, base2, gens2)
    ([1], [(4)(1 2)])
    """
    s = 2 if signed else 0
    n1 = gens1[0].size - s
    base = list(base1)
    base += [x + n1 for x in base2]
    gens1 = [h._array_form for h in gens1]
    gens2 = [h._array_form for h in gens2]
    gens = perm_af_direct_product(gens1, gens2, signed)
    size = len(gens[0])
    id_af = list(range(size))
    gens = [h for h in gens if h != id_af]
    if not gens:
        gens = [id_af]
    return (base, [_af_new(h) for h in gens])