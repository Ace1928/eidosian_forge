from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def coincidence_c(self, alpha, beta):
    """
        A variation of ``coincidence`` routine used in the coset-table based
        method of coset enumeration. The only difference being on addition of
        a new coset in coset table(i.e new coset introduction), then it is
        appended to ``deduction_stack``.

        See Also
        ========

        coincidence

        """
    A_dict = self.A_dict
    A_dict_inv = self.A_dict_inv
    table = self.table
    q = []
    self.merge(alpha, beta, q)
    while len(q) > 0:
        gamma = q.pop(0)
        for x in A_dict:
            delta = table[gamma][A_dict[x]]
            if delta is not None:
                table[delta][A_dict_inv[x]] = None
                self.deduction_stack.append((delta, x ** (-1)))
                mu = self.rep(gamma)
                nu = self.rep(delta)
                if table[mu][A_dict[x]] is not None:
                    self.merge(nu, table[mu][A_dict[x]], q)
                elif table[nu][A_dict_inv[x]] is not None:
                    self.merge(mu, table[nu][A_dict_inv[x]], q)
                else:
                    table[mu][A_dict[x]] = nu
                    table[nu][A_dict_inv[x]] = mu