from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def process_deductions_check(self, R_c_x, R_c_x_inv):
    """
        A variation of ``process_deductions``, this calls ``scan_check``
        wherever ``process_deductions`` calls ``scan``, described on Pg. [1].

        See Also
        ========

        process_deductions

        """
    table = self.table
    while len(self.deduction_stack) > 0:
        alpha, x = self.deduction_stack.pop()
        for w in R_c_x:
            if not self.scan_check(alpha, w):
                return False
        beta = table[alpha][self.A_dict[x]]
        if beta is not None:
            for w in R_c_x_inv:
                if not self.scan_check(beta, w):
                    return False
    return True