from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def coset_representative(self, coset):
    """
        Compute the coset representative of a given coset.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r
        >>> F, x, y = free_group("x, y")
        >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
        >>> C = coset_enumeration_r(f, [x])
        >>> C.compress()
        >>> C.table
        [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]
        >>> C.coset_representative(0)
        <identity>
        >>> C.coset_representative(1)
        y
        >>> C.coset_representative(2)
        y**-1

        """
    for x in self.A:
        gamma = self.table[coset][self.A_dict[x]]
        if coset == 0:
            return self.fp_group.identity
        if gamma < coset:
            return self.coset_representative(gamma) * x ** (-1)