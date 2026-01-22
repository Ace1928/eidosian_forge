from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def look_ahead(self):
    """
        When combined with the HLT method this is known as HLT+Lookahead
        method of coset enumeration, described on pg. 164 [1]. Whenever
        ``define`` aborts due to lack of space available this procedure is
        executed. This routine helps in recovering space resulting from
        "coincidence" of cosets.

        """
    R = self.fp_group.relators
    p = self.p
    for beta in self.omega:
        for w in R:
            self.scan(beta, w)
            if p[beta] < beta:
                break