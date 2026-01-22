from sympy.core import Function, S, Mul, Pow, Add
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.function import expand_func
from sympy.core.symbol import Dummy
from sympy.functions import gamma, sqrt, sin
from sympy.polys import factor, cancel
from sympy.utilities.iterables import sift, uniq
def find_fuzzy(l, x):
    if not l:
        return
    S1, T1 = compute_ST(x)
    for y in l:
        S2, T2 = inv[y]
        if T1 != T2 or (not S1.intersection(S2) and (S1 != set() or S2 != set())):
            continue
        a = len(cancel(x / y).free_symbols)
        b = len(x.free_symbols)
        c = len(y.free_symbols)
        if a == 0 and (b > 0 or c > 0):
            return y