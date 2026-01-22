from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def modified_scan(self, alpha, w, y, fill=False):
    """
        Parameters
        ==========
        \\alpha \\in \\Omega
        w \\in A*
        y \\in (YUY^-1)
        fill -- `modified_scan_and_fill` when set to True.

        See Also
        ========

        scan
        """
    self.scan(alpha, w, y=y, fill=fill, modified=True)