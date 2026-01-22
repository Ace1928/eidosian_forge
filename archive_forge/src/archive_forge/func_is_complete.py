from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def is_complete(self):
    """
        The coset table is called complete if it has no undefined entries
        on the live cosets; that is, `\\alpha^x` is defined for all
        `\\alpha \\in \\Omega` and `x \\in A`.

        """
    return not any((None in self.table[coset] for coset in self.omega))