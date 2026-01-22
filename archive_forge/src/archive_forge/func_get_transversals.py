from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def get_transversals(base, gens):
    """
    Return transversals for the group with BSGS base, gens
    """
    if not base:
        return []
    stabs = _distribute_gens_by_base(base, gens)
    orbits, transversals = _orbits_transversals_from_bsgs(base, stabs)
    transversals = [{x: h._array_form for x, h in y.items()} for y in transversals]
    return transversals