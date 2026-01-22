import itertools
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation
from sympy.combinatorics.free_groups import FreeGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.numbers import igcd
from sympy.ntheory.factor_ import totient
from sympy.core.singleton import S
def orbit_homomorphism(group, omega):
    """
    Return the homomorphism induced by the action of the permutation
    group ``group`` on the set ``omega`` that is closed under the action.

    """
    from sympy.combinatorics import Permutation
    from sympy.combinatorics.named_groups import SymmetricGroup
    codomain = SymmetricGroup(len(omega))
    identity = codomain.identity
    omega = list(omega)
    images = {g: identity * Permutation([omega.index(o ^ g) for o in omega]) for g in group.generators}
    group._schreier_sims(base=omega)
    H = GroupHomomorphism(group, codomain, images)
    if len(group.basic_stabilizers) > len(omega):
        H._kernel = group.basic_stabilizers[len(omega)]
    else:
        H._kernel = PermutationGroup([group.identity])
    return H