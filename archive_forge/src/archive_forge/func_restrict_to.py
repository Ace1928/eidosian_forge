import itertools
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation
from sympy.combinatorics.free_groups import FreeGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.numbers import igcd
from sympy.ntheory.factor_ import totient
from sympy.core.singleton import S
def restrict_to(self, H):
    """
        Return the restriction of the homomorphism to the subgroup `H`
        of the domain.

        """
    if not isinstance(H, PermutationGroup) or not H.is_subgroup(self.domain):
        raise ValueError('Given H is not a subgroup of the domain')
    domain = H
    images = {g: self(g) for g in H.generators}
    return GroupHomomorphism(domain, self.codomain, images)