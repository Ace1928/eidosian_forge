from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def modified_coincidence(self, alpha, beta, w):
    """
        Parameters
        ==========

        A coincident pair `\\alpha, \\beta \\in \\Omega, w \\in Y \\cup Y^{-1}`

        See Also
        ========

        coincidence

        """
    self.coincidence(alpha, beta, w=w, modified=True)