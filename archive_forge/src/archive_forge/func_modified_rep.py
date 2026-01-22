from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def modified_rep(self, k):
    """
        Parameters
        ==========

        `k \\in [0 \\ldots n-1]`

        See Also
        ========

        rep
        """
    self.rep(k, modified=True)