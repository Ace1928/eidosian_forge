from collections import OrderedDict
from ... import sage_helper
def first_pair_differing_in_first_component(L):
    for i in range(len(L)):
        a, b = L[i:i + 2]
        if a[0] != b[0]:
            return (a, b)