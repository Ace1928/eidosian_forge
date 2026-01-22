from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def perm_af_direct_product(gens1, gens2, signed=True):
    """
    Direct products of the generators gens1 and gens2.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import perm_af_direct_product
    >>> gens1 = [[1, 0, 2, 3], [0, 1, 3, 2]]
    >>> gens2 = [[1, 0]]
    >>> perm_af_direct_product(gens1, gens2, False)
    [[1, 0, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5], [0, 1, 2, 3, 5, 4]]
    >>> gens1 = [[1, 0, 2, 3, 5, 4], [0, 1, 3, 2, 4, 5]]
    >>> gens2 = [[1, 0, 2, 3]]
    >>> perm_af_direct_product(gens1, gens2, True)
    [[1, 0, 2, 3, 4, 5, 7, 6], [0, 1, 3, 2, 4, 5, 6, 7], [0, 1, 2, 3, 5, 4, 6, 7]]
    """
    gens1 = [list(x) for x in gens1]
    gens2 = [list(x) for x in gens2]
    s = 2 if signed else 0
    n1 = len(gens1[0]) - s
    n2 = len(gens2[0]) - s
    start = list(range(n1))
    end = list(range(n1, n1 + n2))
    if signed:
        gens1 = [gen[:-2] + end + [gen[-2] + n2, gen[-1] + n2] for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]
    else:
        gens1 = [gen + end for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]
    res = gens1 + gens2
    return res