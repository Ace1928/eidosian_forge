from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def coincidence(self, alpha, beta, w=None, modified=False):
    """
        The third situation described in ``scan`` routine is handled by this
        routine, described on Pg. 156-161 [1].

        The unfortunate situation when the scan completes but not correctly,
        then ``coincidence`` routine is run. i.e when for some `i` with
        `1 \\le i \\le r+1`, we have `w=st` with `s = x_1 x_2 \\dots x_{i-1}`,
        `t = x_i x_{i+1} \\dots x_r`, and `\\beta = \\alpha^s` and
        `\\gamma = \\alpha^{t-1}` are defined but unequal. This means that
        `\\beta` and `\\gamma` represent the same coset of `H` in `G`. Described
        on Pg. 156 [1]. ``rep``

        See Also
        ========

        scan

        """
    A_dict = self.A_dict
    A_dict_inv = self.A_dict_inv
    table = self.table
    q = []
    if modified:
        self.modified_merge(alpha, beta, w, q)
    else:
        self.merge(alpha, beta, q)
    while len(q) > 0:
        gamma = q.pop(0)
        for x in A_dict:
            delta = table[gamma][A_dict[x]]
            if delta is not None:
                table[delta][A_dict_inv[x]] = None
                mu = self.rep(gamma, modified=modified)
                nu = self.rep(delta, modified=modified)
                if table[mu][A_dict[x]] is not None:
                    if modified:
                        v = self.p_p[delta] ** (-1) * self.P[gamma][self.A_dict[x]] ** (-1)
                        v = v * self.p_p[gamma] * self.P[mu][self.A_dict[x]]
                        self.modified_merge(nu, table[mu][self.A_dict[x]], v, q)
                    else:
                        self.merge(nu, table[mu][A_dict[x]], q)
                elif table[nu][A_dict_inv[x]] is not None:
                    if modified:
                        v = self.p_p[gamma] ** (-1) * self.P[gamma][self.A_dict[x]]
                        v = v * self.p_p[delta] * self.P[mu][self.A_dict_inv[x]]
                        self.modified_merge(mu, table[nu][self.A_dict_inv[x]], v, q)
                    else:
                        self.merge(mu, table[nu][A_dict_inv[x]], q)
                else:
                    table[mu][A_dict[x]] = nu
                    table[nu][A_dict_inv[x]] = mu
                    if modified:
                        v = self.p_p[gamma] ** (-1) * self.P[gamma][self.A_dict[x]] * self.p_p[delta]
                        self.P[mu][self.A_dict[x]] = v
                        self.P[nu][self.A_dict_inv[x]] = v ** (-1)