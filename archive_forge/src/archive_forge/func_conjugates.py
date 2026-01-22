from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def conjugates(self, R):
    R_c = list(chain.from_iterable(((rel.cyclic_conjugates(), (rel ** (-1)).cyclic_conjugates()) for rel in R)))
    R_set = set()
    for conjugate in R_c:
        R_set = R_set.union(conjugate)
    R_c_list = []
    for x in self.A:
        r = {word for word in R_set if word[0] == x}
        R_c_list.append(r)
        R_set.difference_update(r)
    return R_c_list