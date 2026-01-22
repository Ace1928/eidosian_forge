from sympy.ntheory.primetest import isprime
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.printing.defaults import DefaultPrinting
from sympy.combinatorics.free_groups import free_group
def leading_exponent(self, element):
    """
        Return the leading non-zero exponent.

        Explanation
        ===========

        The leading exponent for a given element `g` is defined
        by `\\mathrm{leading\\_exponent}[g]` `= e_i`, if `\\mathrm{depth}[g] = i`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> collector.leading_exponent(G[1])
        1

        """
    exp_vector = self.exponent_vector(element)
    depth = self.depth(element)
    if depth != len(self.pcgs) + 1:
        return exp_vector[depth - 1]
    return None