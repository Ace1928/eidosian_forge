from functools import reduce
from sympy.plotting.intervalmath import interval
from sympy.external import import_module
def reduce_and(cmp_intervala, cmp_intervalb):
    if cmp_intervala[0] is False or cmp_intervalb[0] is False:
        first = False
    elif cmp_intervala[0] is None or cmp_intervalb[0] is None:
        first = None
    else:
        first = True
    if cmp_intervala[1] is False or cmp_intervalb[1] is False:
        second = False
    elif cmp_intervala[1] is None or cmp_intervalb[1] is None:
        second = None
    else:
        second = True
    return (first, second)